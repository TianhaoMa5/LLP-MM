import math, hashlib, sys, json
import numpy as np
import torch
import torch.nn.functional as F


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


def accuracy(network, loader, device):
    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            batch_weights = torch.ones(len(x))
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (
                    (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                )
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def _unpack_validation_bags(var1, var2, num_classes, device):
    """Flatten a legacy validation batch without assuming equal bag sizes."""
    bag_container = var1[0]
    number_of_bags = len(var2[0])
    if len(bag_container) != number_of_bags:
        raise ValueError(
            f"validation inputs contain {len(bag_container)} bags but "
            f"proportions contain {number_of_bags}"
        )

    bags = [bag_container[index] for index in range(number_of_bags)]
    sizes = torch.as_tensor([len(bag) for bag in bags], dtype=torch.long, device=device)
    if bool((sizes <= 0).any()):
        raise ValueError("validation contains an empty bag")
    images = torch.cat(bags, dim=0).to(device)
    proportions = torch.tensor(
        [
            [var2[class_index][bag_index] for class_index in range(num_classes)]
            for bag_index in range(number_of_bags)
        ],
        dtype=torch.float64,
        device=device,
    )
    bag_index = torch.repeat_interleave(
        torch.arange(number_of_bags, device=device), sizes
    )
    return images, proportions, sizes, bag_index


def val_PM(args, network, loader, device):
    network.eval()

    total_dist = 0.0
    total_bags = 0

    with torch.no_grad():
        val_minibatches_iterator = iter(loader)
        for it in range(len(val_minibatches_iterator)):
            (var1, var2, var3, var4, var5) = next(val_minibatches_iterator)
            var1 = var1[0]
            length = len(var2[0])
            C = args.n_classes

            imsw = []
            for i in range(length):
                imsw.append(var1[i])
            ims_u_weak = torch.cat(imsw, dim=0)

            imgs = ims_u_weak.to(device)

            true_prop = []
            for i in range(length):
                true_prop.append([var2[j][i] for j in range(C)])
            true_prop = torch.tensor(true_prop, dtype=torch.float32, device=device)

            p = network.predict(imgs)
            if p.dim() != 2 or p.size(1) != C:
                raise ValueError(
                    f"predict output should be [N,C], got {tuple(p.shape)}"
                )

            pred_cls = p.argmax(dim=1)
            pred_oh = F.one_hot(pred_cls, num_classes=C).float()

            N = pred_oh.size(0)
            if N % length:
                raise ValueError(f"N={N} is not divisible by bag count={length}")
            m = N // length
            expected = length * m
            if N != expected:
                raise ValueError(
                    f"N={N} != length*m={expected}, check args.bagsize or concat logic."
                )

            pred_prop = pred_oh.view(length, m, C).mean(dim=1)

            per_bag_dist = (pred_prop - true_prop).abs().sum(dim=1) / C

            total_dist += per_bag_dist.sum().item()
            total_bags += length

    network.train()

    return total_dist / max(total_bags, 1)


def val_DSQ(args, network, loader, device):
    network.eval()

    term1_sum = 0.0
    total_bags = 0
    total_instances = 0
    sum_k_minus_one = 0.0
    prediction_sum = None
    target_count_sum = None

    with torch.no_grad():
        val_minibatches_iterator = iter(loader)
        for it in range(len(val_minibatches_iterator)):
            (var1, var2, var3, var4, var5) = next(val_minibatches_iterator)
            C = args.n_classes
            imgs, true_prop, sizes_long, bag_index = _unpack_validation_bags(
                var1, var2, C, device
            )

            p = network.predict(imgs)
            if p.dim() != 2 or p.size(1) != C:
                raise ValueError(
                    f"predict output should be [N,C], got {tuple(p.shape)}"
                )

            pred_cls = p.argmax(dim=1)
            pred_oh = F.one_hot(pred_cls, num_classes=C).to(
                dtype=torch.float32, device=device
            )

            number_of_bags = len(sizes_long)
            sizes = sizes_long.to(pred_oh)
            bag_prediction_sums = torch.zeros(
                (number_of_bags, C), dtype=pred_oh.dtype, device=device
            )
            bag_prediction_sums.scatter_add_(
                0, bag_index[:, None].expand_as(pred_oh), pred_oh
            )
            predicted_proportions = bag_prediction_sums / sizes[:, None]
            true_prop = true_prop.to(pred_oh)

            per_bag_mse = (predicted_proportions - true_prop).pow(2).mean(dim=1)
            term1_sum += float((sizes * per_bag_mse).sum().item())
            total_bags += number_of_bags
            total_instances += int(sizes_long.sum().item())
            sum_k_minus_one += float((sizes - 1.0).sum().item())

            current_prediction_sum = pred_oh.sum(dim=0)
            current_target_count_sum = (sizes[:, None] * true_prop).sum(dim=0)
            prediction_sum = (
                current_prediction_sum
                if prediction_sum is None
                else prediction_sum + current_prediction_sum
            )
            target_count_sum = (
                current_target_count_sum
                if target_count_sum is None
                else target_count_sum + current_target_count_sum
            )

    network.train()
    if total_bags == 0 or total_instances == 0:
        raise ValueError("DSQ validation loader contains no non-empty bags")

    term1 = term1_sum / total_bags
    mean_k_minus_one = sum_k_minus_one / total_bags
    global_pred_mean = prediction_sum / total_instances
    global_true_mean = target_count_sum / total_instances
    correction = (global_pred_mean - global_true_mean).pow(2).mean()
    dsq_loss = term1 - mean_k_minus_one * float(correction.item())
    return dsq_loss * args.n_classes / 2


def val_easy(args, prior_all, network, loader, device):
    network.eval()

    total_loss = 0.0
    total_instances = 0
    prior = torch.as_tensor(prior_all, dtype=torch.float64, device=device)
    if prior.dim() != 1 or prior.numel() != args.n_classes:
        raise ValueError(
            f"EasyLLP evaluation prior must have shape [{args.n_classes}], "
            f"got {tuple(prior.shape)}"
        )

    with torch.no_grad():
        val_minibatches_iterator = iter(loader)
        for it in range(len(val_minibatches_iterator)):
            (var1, var2, var3, var4, var5) = next(val_minibatches_iterator)
            imgs, proportion, sizes_long, bag_index = _unpack_validation_bags(
                var1, var2, args.n_classes, device
            )
            sizes = sizes_long.to(proportion)
            bag_weight = sizes[:, None] * proportion - (sizes[:, None] - 1.0) * prior
            instance_weight = bag_weight[bag_index]

            p = network.predict(imgs)  # [N_instances, C]
            pred_cls = p.argmax(dim=1)  # [N_instances]
            pred_oh = 1 - F.one_hot(pred_cls, num_classes=args.n_classes).to(
                instance_weight
            )

            easy_acc_inst = (instance_weight * pred_oh).sum(dim=1)  # [N_instances]
            total_loss += easy_acc_inst.sum().item()
            total_instances += easy_acc_inst.numel()

    network.train()
    if total_instances == 0:
        raise ValueError("EasyLLP validation loader contains no non-empty bags")
    return total_loss / total_instances


def val_generalUPM(args, network, loader, device):
    """
    GeneralUPM-style Eq.(28) evaluator, but replace CE loss ℓ with 0-1 loss:
        ℓ01(i,r) = 1{argmax != r}

    Return:
        mean_bag_estimator over all bags in val loader
        (same style as val_easy): total_dist / total_bags
    """
    network.eval()
    total_dist = 0.0
    total_bags = 0
    C = int(args.n_classes)

    with torch.no_grad():
        val_iter = iter(loader)
        for _ in range(len(val_iter)):
            (var1, var2, var3, var4, var5) = next(val_iter)

            # ----- build imgs exactly like your val_easy -----
            x_list = var1[0]  # list length = num_bags in batch (you call it length)
            length = len(var2[0])  # num_bags in batch

            imsw = []
            for i in range(length):
                imsw.append(x_list[i])
            ims_u_weak = torch.cat(imsw, dim=0)  # [N, ...], N = length * bagsize

            imgs = ims_u_weak.to(device)

            # ----- build proportions [B,C] exactly like your val_easy -----
            label_proportions = [[] for _ in range(length)]
            for i in range(length):
                labels = []
                for j in range(C):
                    labels.append(var2[j][i])
                label_proportions[i].append(labels)

            proportion = []
            for i in range(length):
                proportion.append(
                    torch.tensor(label_proportions[i][0], dtype=torch.float64).to(
                        device
                    )
                )
            proportions = torch.stack(proportion)  # [B,C] with B=length

            # ----- forward -----
            logits = network.predict(imgs)  # [N,C]
            N = logits.size(0)
            if N % length:
                raise ValueError(f"N={N} is not divisible by bag count={length}")
            m = N // length
            if N % m != 0:
                raise ValueError(f"N={N} must be divisible by bagsize={m}")
            B = N // m
            if proportions.shape != (B, C):
                raise ValueError(
                    f"proportions must be [B,C]=[{B},{C}], got {tuple(proportions.shape)}"
                )

            # ===== replace CE ℓ with 0-1 ℓ01 =====
            pred_cls = logits.argmax(dim=1)  # [N]
            # ell01[i,r] = 1{pred!=r} = 1 - one_hot(pred)
            ell01 = 1.0 - F.one_hot(pred_cls, num_classes=C).to(
                dtype=torch.float64, device=device
            )  # [N,C]

            # ----- Eq.(28) minibatch estimates -----
            p_hat = proportions.mean(dim=0)  # [C]
            E_hat = ell01.mean(dim=0)  # [C]

            ell_bag = ell01.view(B, m, C)  # [B,m,C]
            centered = ell_bag - E_hat.view(1, 1, C)
            sum_centered = centered.sum(dim=1)  # [B,C]

            term1 = ((proportions - p_hat.view(1, C)) * sum_centered).sum(dim=1)  # [B]
            term2 = (p_hat * E_hat).sum()  # scalar

            loss_per_bag = term1 + term2  # [B]

            total_dist += loss_per_bag.sum().item()
            total_bags += B

    network.train()
    return total_dist / max(total_bags, 1)
