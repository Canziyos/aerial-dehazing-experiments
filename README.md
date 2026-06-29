# Aerial Dehazing Experiments

This repository extends the original [Nonhomogeneous Image Dehazing](https://github.com/diptamath/Nonhomogeneous_Image_Dehazing.git) implementation by adapting the training and data workflow for aerial image dehazing experiments using synthetic haze generation and VSAI-style aerial imagery.

The project builds on the Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing, accepted at the NTIRE Workshop, CVPR 2020.

Preprint: https://arxiv.org/abs/2005.05999

## What Changed

- Retrained the DMPHN-based workflow for aerial image dehazing experiments.
- Added synthetic haze generation for clean aerial imagery.
- Added data preparation helpers for train, validation, and test file lists.
- Added model conversion, quantization, pruning, and Qualcomm AI Hub deployment experiments.

## Repository Layout

```text
src/dehazing/
  models.py
  datasets.py
  loss.py

scripts/
  train_dmphn.py
  test_dmphn.py
  train_dmshn.py
  test_dmshn.py
  apply_haze.py
  apply_haze_test.py
  create_txt.py
  prepare_image_data.py

edge/
  convert.py
  compile.py
  inference.py
  quantize_and_profile_test.py
  dmphn_dynamic_quantize.py
  dmphn_prune_quant_export.py

assets/
dataset/
new_dataset/
examples/
```

`src/dehazing/` contains reusable model code. `scripts/` contains training, testing, and data preparation entry points. `edge/` contains conversion, quantization, pruning, profiling, and deployment experiments.

The `new_dataset/val/` folder contains a tiny validation set that can be used as a runnable demo input.

Model checkpoints are intentionally ignored by Git. Keep local DMPHN weights under `checkpoints/`, and keep optional DMSHN weights under `checkpoints2/` if you need to run the DMSHN scripts. Publish larger checkpoint files through GitHub Releases or another external storage location.

## Setup

Install the core dependencies:

```powershell
pip install -r requirements.txt
```

Optional dependencies for edge deployment and compression experiments:

```powershell
pip install -r requirements-edge.txt
```

## Running Inference

Place DMPHN checkpoint files under:

```text
checkpoints/DMPHN_1_2_4/
```

Expected files:

```text
encoder_lv1.pkl
encoder_lv2.pkl
encoder_lv3.pkl
decoder_lv1.pkl
decoder_lv2.pkl
decoder_lv3.pkl
```

Run:

```powershell
python scripts\test_dmphn.py
```

For the DMSHN variant, place the matching local weights under `checkpoints2/DMSHN_1_2_4/`, then run:

```powershell
python scripts\test_dmshn.py
```

## Training

For training, image paths for train, validation, and test data should be listed in text files. For example, patch-level training expects hazy and ground-truth image paths in files such as:

```text
new_dataset/train_patch_hazy.txt
new_dataset/train_patch_gt.txt
```

Run:

```powershell
python scripts\train_dmphn.py
python scripts\train_dmshn.py
```

## Edge Experiments

The edge scripts expect local model artifacts such as checkpoints or `dmphn_dehazing.onnx` depending on the workflow:

```powershell
python edge\convert.py
python edge\compile.py
python edge\inference.py
python edge\quantize_and_profile_test.py
```

These scripts may require the optional packages in `requirements-edge.txt`.

## Results From Original Paper

Quantitative results:

<img src="assets/cvpr_2.png" width="500"/>

Qualitative results:

![](assets/cvpr_1.png)

## Citation

Please cite the original paper if this project helps your research:

```bibtex
@InProceedings{Das_fast_deep_2020,
  author = {Sourya Dipta Das and Saikat Dutta},
  title = {Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2020}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
