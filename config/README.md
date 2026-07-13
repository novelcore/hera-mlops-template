# config/ — your experiment configuration

This tree IS the submit form. Every scalar value in it becomes a form
field (named by its path: `train.epochs` → `train-epochs`); every
directory of option files (like `train/optimizer/`) becomes a dropdown
that swaps a whole subtree at once.

- Add a parameter = add a leaf here. Nothing else to wire.
- Add a preset = add a file to a group directory.
- `config.yaml` lists which group option is the default.
