# ML Downscaling for S2S Forecasts

This repository provides scripts for **preprocessing and downscaling subseasonal-to-seasonal (S2S) forecasts** using deep learning models.  
It supports multiple architectures (CNN, UNet, SRGAN-like generator), seasonal-aware train/test splitting, cross-validation, and model evaluation/plotting.

---

## Repository Structure

- **`prepData.py`** — Data preparation & preprocessing  
  - Handles regridding between low- and high-resolution datasets  
  - Applies seasonal train/test splits or k-fold cross-validation  
  - Prepares PyTorch datasets for model training  
  - Includes utilities for plotting HR vs LR comparisons  

- **`dscale.py`** — Model training & evaluation  
  - Defines CNN, U-Net, and SRGAN-like generator downscaling models  
  - Training and testing routines with configurable epochs & batch size  
  - Evaluates models using **MAE, RMSE, and Pearson correlation**  
  - Saves plots and NetCDF outputs with predictions and RMSE fields  

---

## Environment Setup

Python ≥ 3.9 is recommended. Install dependencies with:

```bash
pip install torch torchvision xarray numpy pandas matplotlib seaborn xesmf tqdm scikit-learn
```

---

## Workflow

### 1. Data Preparation (`prepData.py`)
- **Regridding:** Aligns low-resolution forecasts with high-resolution reference grids using `xESMF`.  
- **Splitting:**
  - Seasonal-aware splits (`split_data_for_training_and_testing`)  
  - K-fold cross-validation (`split_data_for_kfold_cross_validation`)  
- **Output:** PyTorch-ready `TensorDataset` objects for training/testing.

Example:
```python
from prepData import regrid, split_data_for_training_and_testing, prepare_data

low_res, high_res = regrid(dsLR, dsHR, var='t2')
splits = split_data_for_training_and_testing(low_res, high_res, random_seed=42)
train_dataset, test_dataset = prepare_data(splits, lr_var='t2', hr_var='t2')
```

---

### 2. Model Training & Evaluation (`dscale.py`)
- **Models implemented:**
  - `CNNDownscale`
  - `UNetDownscale`
  - `SRGANGenerator`
- **Training:**  
  Uses `train_model` with configurable optimizer, scheduler, and MSE loss.  
- **Evaluation:**  
  - Generates plots comparing predictions vs truth and statistical downscaling  
  - Computes **MAE, RMSE, Pearson correlation**  
  - Saves results into NetCDF (`*_predictions_with_rmse.nc`)

Example:
```python
from dscale import CNNDownscale, train_and_test_model
from prepData import prepare_data, regrid, split_data_for_training_and_testing

# Prepare datasets
low_res, high_res = regrid(dsLR, dsHR, 't2')
splits = split_data_for_training_and_testing(low_res, high_res)
train_dataset, test_dataset = prepare_data(splits, lr_var='t2', hr_var='t2')

# Train & evaluate CNN model
cnn_model = CNNDownscale(input_channels=1)
pred, truth = train_and_test_model(cnn_model, 'F', train_dataset, test_dataset, 'cnn_t2', 0, matchedSDS, '12km', 'wk1')
```

---

## Outputs

- **Plots:**  
  Side-by-side visualizations of:
  - Model high-resolution predictions  
  - True HR data  
  - Statistical downscaled HR  
  - Difference maps  

- **NetCDF files:**  
  Saved under `output/{res}/{wk}/DL_models_*_predictions_with_rmse.nc`  
  Includes:
  - Model predictions (`*_predicted`)  
  - RMSE fields (`*_rmse`)  

---

## Example Run

The `__main__` block in `dscale.py` shows an example for week 1 (`wk1`) with NAM 12 km HR data:

```bash
python dscale.py
```

It will:
1. Load and regrid LR/HR datasets
2. Train CNN, UNet, and SRGAN models
3. Evaluate models at monitoring sites
4. Save plots and NetCDF outputs

---

## Notes
- `units` and `varN` dictionaries in `dscale.py` define variable-specific units and names (`t2`, `ws`, `vpd`).  
- Training defaults: `epochs=100`, `batch_size=8`.  
- Distributed training scaffolding (`torch.distributed`) is included but not enabled by default.  

---