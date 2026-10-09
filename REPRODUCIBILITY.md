# Reproducing the paper

The code covers image benchmarks, order ablations, patient-grouped experiments,
and GAN comparisons. Commands are listed in [README](README.md).

## Experiment settings

[paper_protocol.json](configs/paper_protocol.json) lists the experiment matrix.
`mo_matching.core.hparams_registry` defines executable defaults; `mo_matching.run`
resolves method aliases, ABS, dataset, order, and batch settings into a single
training command. Every run saves resolved arguments and hyperparameters.

- Image tables: modified-stem ResNet-18 for CIFAR and ResNet-34 for miniImageNet;
  500 epochs, 1024 images/update; SGD with momentum 0.9 and weight decay 0.0005;
  learning rate 0.005 for PVC and 0.05 otherwise. Linear warmup covers 8% of
  updates, followed by quarter-cosine decay. FlowLLP uses Nesterov momentum
  for image experiments.
- MM: fixed-image `paper_image` kernel; orders 8/3/3, equal weights and CE.
  KU uses its separate float64 `stable_dp` kernel with smoothing 0.0001.
- KU: standard-stem pretrained ResNet-18, Adam at 0.001, 100 epochs, five warmup
  epochs, four complete bags/update (five for GeneralUPM and GeneralUPM-ABS),
  forward chunks of 32 images, and orders 3/5/8 over three seeds.
- GAN: shared bags and models, 500 epochs, discriminator SGD, generator Adam at
  0.0003 with betas 0.5/0.999, noise dimension 100, unit loss weights. The discriminator uses five
  warmup epochs followed by standard cosine decay. MM+GAN
  changes only the proportion objective to uniform order-eight MM. AMP and
  discriminator Nesterov are disabled by the paper runner.

## Reporting

Image tables use the highest recorded finite **test accuracy** for each seed,
then mean and sample standard deviation. KU selects the highest **test Macro-F1**
record and reports all four metrics at that same checkpoint, then mean and
population standard deviation. GAN uses the highest recorded finite test accuracy
and sample standard deviation. For numerically divergent runs, use the highest
finite test accuracy recorded before divergence.

## Verification

CI checks the moment objectives, gradients, bag assignments, baseline updates,
and training entry points. Smoke runs verify setup and are excluded from formal
results.
