# Aerial Dehazing Experiments

Clean rebuild of the aerial dehazing pipeline around two rebuilt model variants:

- `DMPHN124` — Deep Multi-Patch Hierarchical Network, 1-2-4 variant.
- `DMSHN124` — Deep Multi-Scale Hierarchical Network, 1-2-4 variant.

The repository keeps model architecture, data loading, training, inference, evaluation, and scripts separated.

## Layout

```text
aerial-dehazing-experiments/
  dehazing/
    architectures/
    data/
    evaluation/
    inference/
    losses/
    modules/
    training/
  scripts/
  docs/
  dataset/
  new_dataset/
  tmp_dataset/
```

## Important tensor convention

- Dataset tensors are loaded in `[0, 1]`.
- Training and inference subtract `0.5` before model input.
- Model output is in model-domain space and is not guaranteed to be image-range.
- Inference adds `0.5`, clamps to `[0, 1]`, and saves the result.

## Main commands

Print model information:

```powershell
python -m scripts.model_info --model DMPHN124
python -m scripts.model_info --model DMSHN124
```

Run the smoke training test:

```powershell
python -m scripts.testing_train
```

Inspect paired dataset lists:

```powershell
python -m scripts.inspect_dataset --root-dir dataset --hazy-list dataset/hazy.txt --clean-list dataset/GT.txt --max-check 20
```

Train with the generic entry point:

```powershell
python -m scripts.train --model DMPHN124 --root-dir dataset --hazy-list dataset/hazy.txt --clean-list dataset/GT.txt --val-hazy-list dataset/val_hazy.txt --val-clean-list dataset/val_GT.txt --epochs 1 --batch-size 1 --device cpu --torch-threads 1 --output-dir outputs/dmphn
```

The model-specific wrappers still work:

```powershell
python -m scripts.train_dmphn --root-dir dataset --hazy-list dataset/hazy.txt --clean-list dataset/GT.txt --epochs 1 --batch-size 1 --device cpu --torch-threads 1 --output-dir outputs/dmphn
python -m scripts.train_dmshn --root-dir dataset --hazy-list dataset/hazy.txt --clean-list dataset/GT.txt --epochs 1 --batch-size 1 --device cpu --torch-threads 1 --output-dir outputs/dmshn
```

Convert original six-file DMPHN checkpoints to one clean checkpoint:

```powershell
python -m scripts.convert_legacy_dmphn_checkpoint --legacy-dir checkpoints/DMPHN_1_2_4 --output outputs/dmphn_legacy_converted/checkpoint.pt --device cpu
```

More commands are in [`docs/commands.md`](docs/commands.md).

## Status

The engineering pipeline is functional: model construction, smoke training, checkpoint save/load, inference, folder inference, PSNR evaluation, and legacy DMPHN checkpoint conversion are supported.

Actual output quality depends on real training. One dummy image for one epoch only proves the plumbing works; it does not prove dehazing quality. Obviously. 😄

## Original project context

This project was derived from the Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing, NTIRE Workshop, CVPR 2020.

Preprint: https://arxiv.org/abs/2005.05999

Please cite the original work when appropriate:

```bibtex
@InProceedings{Das_fast_deep_2020,
  author = {Sourya Dipta Das and Saikat Dutta},
  title = {Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2020}
}
```
