# Vessel Segmentation Report Context

This file is the compact implementation brief used to generate `main.tex`.

## Source runs
- Training run: W&B `eeebashit/vessel-seg/gsutr3pf`, Explorer job `6670898`.
- Training hardware: NVIDIA A100-SXM4-40GB, CUDA 12.1, PyTorch 2.5.1+cu121.
- Prediction run: local W&B run `run-20260509_162223-qi7txuwi`, Explorer job `6685455`.
- Prediction hardware: Tesla V100-SXM2-32GB, CUDA 12.1, PyTorch 2.5.1+cu121.

## Local report artifacts
- Training media: `figures/wandb/training/` (435 PNG files).
- Prediction media: `figures/wandb/prediction/` (133 PNG files).
- Downsampled report images: `figures/report_images/`.
- Logs and split file: `source_logs/`.
- Training manifest: `figures/manifest_training.csv`.
- Prediction manifest: `figures/manifest_prediction.csv`.

## Validation visualization mapping
- Fold 1: `13_20250724_Plate1_C3_D7_MAX_Crop.tif`, split `validation`, epochs [10, 20, 30, 40, 50, 60, 70], best epoch 48, validation loss 0.0568.
- Fold 2: `42_20250807_Plate1_C2_D21_MAX_Crop.tif`, split `validation`, epochs [10, 20, 30, 40, 50, 60], best epoch 33, validation loss 0.0713.
- Fold 3: `2_20260418_EC_Plate1_A2_D7_MAX_Crop.tif`, split `validation`, epochs [10, 20, 30, 40, 50, 60], best epoch 37, validation loss 0.0627.
- Fold 4: `7_20250724_Plate1_B2_D7_MAX_Crop.tif`, split `validation`, epochs [10, 20, 30, 40, 50], best epoch 23, validation loss 0.0624.
- Fold 5: `18_20250731_Plate1_A3_D14_MAX_Crop.tif`, split `validation`, epochs [10, 20, 30, 40, 50], best epoch 22, validation loss 0.0721.

## Final prediction mapping
- Index 0: `16_20250731_Plate1_A1_D14_MAX_Crop.tif`, split `train/validation pool`, held out from training: no.
- Index 1: `17_20260418_EC_Plate1_A2_D14_MAX_Crop.tif`, split `test`, held out from training: yes, Dice 0.9507.
- Index 2: `20_20260418_EC_Plate2_A2_D14_MAX_Crop.tif`, split `train/validation pool`, held out from training: no.
- Index 3: `32_20250807_Plate1_A2_D21_MAX_Crop.tif`, split `test`, held out from training: yes, Dice 0.9110.
- Index 4: `43_20250807_Plate1_C3_D21_MAX_Crop.tif`, split `test`, held out from training: yes, Dice 0.9141.
- Index 5: `12_20250724_Plate1_C2_D7_MAX_Crop.tif`, split `test`, held out from training: yes, Dice 0.9154.
- Index 6: `1_20250724_Plate1_A1_D7_MAX_Crop.tif`, split `test`, held out from training: yes, Dice 0.9036.

## Metrics reported
- 5-fold cross-validation mean +/- std: 0.0651 +/- 0.0058.
- Selected fold: 1, best epoch 48, validation loss 0.0568.
- Test loss: 0.0662.
- Tile-level test Dice: 0.9158.
- Mean stitched-image Dice: 0.9189.
- Stitched Dice std: 0.0164.
- 95% CI: [0.8962, 0.9417].
- All 5 held-out test images reached Dice >= 0.90.
