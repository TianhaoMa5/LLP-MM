"""Particle Flow integration for the current PLeNCH training interface."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    import ot
except ImportError:  # pragma: no cover - exercised by the explicit error path
    ot = None


def _squared_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_norm = first.pow(2).sum(dim=1, keepdim=True)
    second_norm = second.pow(2).sum(dim=1, keepdim=True)
    return (first_norm + second_norm.t() - 2 * first @ second.t()).clamp_min_(0)


def build_flowllp_algorithm(
    base_class,
    scheduler_class,
    resolve_bag_layout,
    bag_means,
):
    """Build against PLeNCH's local base classes without a circular import."""

    class LLP_FlowLLP(base_class):
        """Particle Flow: bag pretraining, particle learning, then joint tuning.

        Natural bags are flattened and described by ``bag_sizes`` and
        ``bag_index``.  The particle objective is unchanged: every cached bag
        has unit source mass distributed uniformly over its actual instances.
        """

        def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
            super().__init__(epochs, input_shape, train_givenY, hparams, bagsize)
            hp = hparams if isinstance(hparams, dict) else vars(hparams)

            self.flow_latent_dim = int(hp.get("flow_latent_dim", 50))
            self.flow_pretrain_fraction = float(
                hp.get("flow_pretrain_fraction", 0.5)
            )
            self.flow_anchors_per_class = int(
                hp.get("flow_anchors_per_class", 1000)
            )
            self.flow_particle_steps = int(hp.get("flow_particle_steps", 3000))
            self.flow_particle_lr = float(hp.get("flow_particle_lr", 1e-3))
            self.flow_anchor_bag_batch = int(hp.get("flow_anchor_bag_batch", 1))
            self.flow_lambda_bag = float(hp.get("flow_lambda_bag", 1.0))
            self.flow_lambda_anchor = float(hp.get("flow_lambda_anchor", 0.1))
            self.flow_reg_label = float(hp.get("flow_reg_label", 0.0))
            self.flow_reg_bag_classifier = float(
                hp.get("flow_reg_bag_classifier", 1.0)
            )
            self.flow_anchor_batch_size = int(
                hp.get("flow_anchor_batch_size", 128)
            )
            self.flow_steps_per_epoch = int(hp.get("steps_per_epoch", 1))
            self.flow_total_steps = max(1, int(epochs))
            self.flow_pretrain_steps = min(
                self.flow_total_steps,
                max(
                    1,
                    int(
                        round(
                            self.flow_total_steps * self.flow_pretrain_fraction
                        )
                    ),
                ),
            )
            self.flow_cache_start = max(
                0, self.flow_pretrain_steps - self.flow_steps_per_epoch
            )

            # The image implementation learns particles in a latent projection
            # space rather than directly in the backbone output space.
            self.projector = nn.Sequential(
                nn.Linear(self.featurizer.n_outputs, self.flow_latent_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.classifier = nn.Linear(self.flow_latent_dim, self.num_classes)
            self.network = nn.Sequential(
                self.featurizer, self.projector, self.classifier
            )
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=hp["lr"],
                momentum=0.9,
                weight_decay=hp.get("weight_decay", 5e-4),
                nesterov=bool(hp.get("nesterov", True)),
            )
            warmup_iter = int(0.08 * self.flow_total_steps)
            warmup_ratio = 5e-5 / hp["lr"]
            self.scheduler = scheduler_class(
                self.optimizer,
                self.flow_total_steps,
                warmup_iter=warmup_iter,
                warmup_ratio=warmup_ratio,
                warmup="linear",
            )

            number_of_anchors = (
                self.num_classes * self.flow_anchors_per_class
            )
            self.register_buffer(
                "flow_anchor_features",
                torch.zeros(number_of_anchors, self.flow_latent_dim),
            )
            self.register_buffer(
                "flow_anchor_labels",
                torch.arange(self.num_classes).repeat_interleave(
                    self.flow_anchors_per_class
                ),
            )
            self.register_buffer(
                "flow_anchors_ready", torch.tensor(False, dtype=torch.bool)
            )
            self.register_buffer(
                "flow_step", torch.tensor(0, dtype=torch.long)
            )
            self._flow_bag_cache: list[dict[str, torch.Tensor]] = []

        def _forward_with_features(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            features = self.projector(self.featurizer(x))
            return features, self.classifier(features)

        def _layout(
            self,
            number_of_instances: int,
            proportions: torch.Tensor,
            bag_sizes: torch.Tensor | None,
            bag_index: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return resolve_bag_layout(
                number_of_instances,
                proportions,
                self.num_classes,
                proportions.device,
                self.bagsize,
                bag_sizes,
                bag_index,
            )

        def _bag_proportion_loss(
            self,
            logits: torch.Tensor,
            proportions: torch.Tensor,
            bag_index: torch.Tensor,
        ) -> torch.Tensor:
            probabilities = F.softmax(logits, dim=1)
            predictions = bag_means(
                probabilities, bag_index, len(proportions)
            ).clamp_min(1e-7)
            return -(proportions * predictions.log()).sum(dim=1).mean()

        @torch.no_grad()
        def _cache_particle_bags(
            self,
            features: torch.Tensor,
            logits: torch.Tensor,
            proportions: torch.Tensor,
            bag_index: torch.Tensor,
        ) -> None:
            for bag_id in range(len(proportions)):
                positions = torch.nonzero(
                    bag_index == bag_id, as_tuple=False
                ).squeeze(1)
                self._flow_bag_cache.append(
                    {
                        "data": features[positions].detach().float().cpu(),
                        "prop": proportions[bag_id].detach().float().cpu(),
                        "y_pred_noisy": logits[positions]
                        .detach()
                        .argmax(dim=1)
                        .cpu(),
                    }
                )

        def _learn_particles(self) -> None:
            if ot is None:
                raise ImportError(
                    "LLP_FlowLLP requires POT; install it with `pip install POT`"
                )
            if not self._flow_bag_cache:
                raise RuntimeError(
                    "Particle Flow has no cached bags. Ensure steps_per_epoch "
                    "is passed by plench.train and pretraining is not skipped."
                )

            device = torch.device("cpu")
            number_of_anchors = len(self.flow_anchor_labels)
            embedding = nn.Embedding(
                number_of_anchors, self.flow_latent_dim, device=device
            )
            embedding.weight.data.normal_(mean=0.0, std=2.0)
            optimizer = torch.optim.Adam(
                embedding.parameters(),
                lr=self.flow_particle_lr,
                betas=(0.9, 0.999),
            )
            anchor_labels = self.flow_anchor_labels.detach().cpu()
            classifier = copy.deepcopy(self.classifier).cpu().eval()
            for parameter in classifier.parameters():
                parameter.requires_grad_(False)

            print(
                "Particle Flow: learning "
                f"{number_of_anchors} anchors from "
                f"{len(self._flow_bag_cache)} bags for "
                f"{self.flow_particle_steps} steps"
            )
            for particle_step in range(self.flow_particle_steps):
                count = min(
                    self.flow_anchor_bag_batch, len(self._flow_bag_cache)
                )
                bag_ids = np.random.permutation(
                    len(self._flow_bag_cache)
                )[:count]
                objective = torch.zeros((), device=device)
                for bag_id in bag_ids:
                    bag = self._flow_bag_cache[int(bag_id)]
                    features = bag["data"].reshape(-1, self.flow_latent_dim)
                    source_mass = torch.full(
                        (len(features),), 1.0 / len(features)
                    )
                    target_mass = (
                        bag["prop"].repeat_interleave(
                            self.flow_anchors_per_class
                        )
                        / self.flow_anchors_per_class
                    )
                    target_mass = target_mass / target_mass.sum()
                    cost = _squared_distance(features, embedding.weight)
                    if self.flow_reg_label > 0 and particle_step > 0:
                        mismatch = 1 - (
                            F.one_hot(
                                bag["y_pred_noisy"], self.num_classes
                            ).float()
                            @ F.one_hot(
                                anchor_labels, self.num_classes
                            ).float().t()
                        )
                        cost = cost + self.flow_reg_label * mismatch
                    with torch.no_grad():
                        transport = ot.emd(
                            source_mass, target_mass, cost.detach()
                        )
                    if not isinstance(transport, torch.Tensor):
                        transport = torch.as_tensor(
                            transport, dtype=cost.dtype, device=cost.device
                        )
                    objective = objective + (cost * transport).sum()
                    if self.flow_reg_bag_classifier:
                        objective = objective + (
                            self.flow_reg_bag_classifier
                            * F.cross_entropy(
                                classifier(embedding.weight), anchor_labels
                            )
                        )

                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
                if (particle_step + 1) % 100 == 0:
                    print(
                        "Particle Flow: "
                        f"step={particle_step + 1}/"
                        f"{self.flow_particle_steps} "
                        f"loss={float(objective.detach()):.6f}"
                    )

            self.flow_anchor_features.copy_(
                embedding.weight.detach().to(
                    self.flow_anchor_features.device
                )
            )
            self.flow_anchors_ready.fill_(True)
            self._flow_bag_cache.clear()

        def _anchor_loss(self) -> torch.Tensor:
            count = len(self.flow_anchor_labels)
            batch_size = min(self.flow_anchor_batch_size, count)
            indices = torch.randint(
                count,
                (batch_size,),
                device=self.flow_anchor_features.device,
            )
            logits = self.classifier(self.flow_anchor_features[indices])
            return F.cross_entropy(logits, self.flow_anchor_labels[indices])

        def update(self, minibatches):
            x, proportions = minibatches
            features, logits = self._forward_with_features(x)
            return self.update_from_outputs(features, logits, proportions)

        def update_from_outputs(
            self,
            features: torch.Tensor,
            logits: torch.Tensor,
            proportions: torch.Tensor,
            bag_sizes: torch.Tensor | None = None,
            bag_index: torch.Tensor | None = None,
        ) -> dict[str, float]:
            if features.dim() != 2 or len(features) != len(logits):
                raise ValueError(
                    "FlowLLP features and logits must have matching [N,*] rows"
                )
            if features.shape[1] != self.flow_latent_dim:
                raise ValueError(
                    f"FlowLLP features must have dimension "
                    f"{self.flow_latent_dim}, got {features.shape[1]}"
                )
            proportions = proportions.to(
                device=logits.device, dtype=logits.dtype
            )
            sizes, index = self._layout(
                len(logits), proportions, bag_sizes, bag_index
            )
            del sizes  # validation is the only use; index defines all slices.
            step = int(self.flow_step.item())

            if step >= self.flow_pretrain_steps and not bool(
                self.flow_anchors_ready.item()
            ):
                self._learn_particles()

            bag_loss = self._bag_proportion_loss(
                logits, proportions, index
            )
            if bool(self.flow_anchors_ready.item()):
                anchor_loss = self._anchor_loss()
                loss = (
                    self.flow_lambda_bag * bag_loss
                    + self.flow_lambda_anchor * anchor_loss
                )
                stage = 2.0
            else:
                anchor_loss = torch.zeros_like(bag_loss)
                loss = bag_loss
                stage = 1.0

            if self.flow_cache_start <= step < self.flow_pretrain_steps:
                self._cache_particle_bags(
                    features, logits, proportions, index
                )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.flow_step.add_(1)
            return {
                "loss": float(loss.detach()),
                "bag_loss": float(bag_loss.detach()),
                "anchor_loss": float(anchor_loss.detach()),
                "flow_stage": stage,
            }

    LLP_FlowLLP.__name__ = "LLP_FlowLLP"
    LLP_FlowLLP.__qualname__ = "LLP_FlowLLP"
    return LLP_FlowLLP
