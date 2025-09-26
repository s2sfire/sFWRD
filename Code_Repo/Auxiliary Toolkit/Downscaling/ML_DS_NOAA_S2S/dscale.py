import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

import glob
import os
import matplotlib.pyplot as plt
from prepData import prepare_data, regrid, split_data_for_training_and_testing, open_files_with_subset, split_data_for_kfold_cross_validation
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm  # For progress bars
import seaborn as sns
import re
import gc

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

world_size = torch.cuda.device_count()  # Number of available GPUs

units = {'t2':'K','ws':'m/s','vpd':'hPa'}
varN = {'t2':'T2','ws':'WS','vpd':'VPD'}

myEpoch = 100
batch_size=8

def setup(rank, world_size):
    """Initialize the process group for distributed training."""
    if dist.is_initialized():  # Prevent multiple initializations
        return
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.init_process_group("gloo",  init_method="tcp://127.0.0.1:29500", rank=rank, world_size=world_size)

def cleanup():
    """Destroy process group after training"""
    dist.destroy_process_group()

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Define CNNDownscale model
class CNNDownscale(nn.Module):
    def __init__(self, input_channels, predictor_channels=None, upscale_factor=4):
        super(CNNDownscale, self).__init__()
        self.use_predictor = predictor_channels is not None
        # Calculate the total number of input channels
        in_channels = input_channels + (predictor_channels if self.use_predictor else 0)
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Final output convolutional layer
        self.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, lr_data, predictor_data=None):
        # Concatenate low-resolution data and predictor data if predictors are used
        if self.use_predictor:
            x = torch.cat([lr_data, predictor_data], dim=1)
        else:
            x = lr_data
        # Apply first convolution and activation
        x = torch.relu(self.conv1(x))
        # Apply second convolution and activation
        x = torch.relu(self.conv2(x))
        # Final convolution to get the output
        x = self.conv_out(x)
        return x

# Define SRGANGenerator model
class SRGANGenerator(nn.Module):
    def __init__(self, input_channels, predictor_channels=None, upscale_factor=4, num_conv_blocks=5, leaky_relu_slope=0.2):
        super(SRGANGenerator, self).__init__()
        self.use_predictor = predictor_channels is not None
        # Calculate the total number of input channels
        in_channels = input_channels + (predictor_channels if self.use_predictor else 0)
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(leaky_relu_slope)
        # Convolutional blocks for deeper feature extraction
        conv_layers = []
        for _ in range(num_conv_blocks):
            conv_layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            conv_layers.append(nn.LeakyReLU(leaky_relu_slope))
        self.conv_blocks = nn.Sequential(*conv_layers)
        # Final output convolutional layer
        self.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, lr_data, predictor_data=None):
        # Concatenate low-resolution data and predictor data if predictors are used
        if self.use_predictor:
            x = torch.cat([lr_data, predictor_data], dim=1)
        else:
            x = lr_data
        # Apply first convolution and activation
        x = self.relu(self.conv1(x))
        # Apply the convolutional blocks
        x = self.conv_blocks(x)
        # Final convolution to get the output
        x = self.conv_out(x)
        return x

# Define UNetDownscale model
class UNetDownscale(nn.Module):
    def __init__(self, input_channels, predictor_channels=None, upscale_factor=4):
        super(UNetDownscale, self).__init__()
        self.use_predictor = predictor_channels is not None
        # Calculate the total number of input channels
        in_channels = input_channels + (predictor_channels if self.use_predictor else 0)
        # Encoder layers
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Pooling layer to downsample the feature map
        self.pool = nn.MaxPool2d(2)  # Removed ceil_mode to keep dimensions consistent
        # Upsampling layer (transpose convolution) to increase resolution
        self.upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Fixed stride to match pooling
        # Decoder layer
        self.dec1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # Final output layer
        self.out_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, lr_data, predictor_data=None):
        # Concatenate low-resolution data and predictor data if predictors are used
        if self.use_predictor:
            x = torch.cat([lr_data, predictor_data], dim=1)
        else:
            x = lr_data

        # Encoder path
        enc1 = torch.relu(self.enc1(x))
        enc2 = torch.relu(self.enc2(self.pool(enc1)))

        # Decoder path with skip connection
        upconv = self.upconv(enc2)
        upconv = F.interpolate(upconv, size=enc1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate the upsampled feature map with the corresponding encoder feature map
        concat = torch.cat([upconv, enc1], dim=1)
        # Apply the decoder layer
        dec1 = torch.relu(self.dec1(concat))
        
        # Final output layer
        return self.out_conv(dec1)

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input, target)

# Generic training function
def train_model(model, train_dataset, epochs, lr=0.001):
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    #criterion = PerceptualLoss()  # Use perceptual loss for training
    criterion = MSELoss()  # Use perceptual loss for training
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimizer for training the model
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Learning rate scheduler

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (lr_batch, hr_batch) in tqdm(enumerate(train_loader),desc=f"Epoch {epoch+1}"):
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(lr_batch.unsqueeze(1))  # Forward pass, add channel dimension
            loss = criterion(outputs, hr_batch.unsqueeze(1))  # Calculate the loss, add channel dimension to match output
            loss.backward()  # Backward pass to calculate gradients
            optimizer.step()  # Update model parameters
            running_loss += loss.item()
            # Print average loss every 10 batches
            if i % 10 == 9:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0
        scheduler.step()  # Update the learning rate
    print('Finished Training')

# Function to train and test a model with different NaN strategies
def train_and_test_model(model, strategy, train_dataset, test_dataset, model_name, fold, matchedSDS, res, wk):
    model = model.to(device) # Move the model to the GPU if available
    train_model(model, train_dataset, epochs=myEpoch)

    """Run training with multiple GPUs using torch.multiprocessing.spawn()."""
    #mp.spawn(train_model, args=(world_size, model, train_dataset, myEpoch, 0.001), nprocs=world_size, join=True)
    #train_model(1,1 , model, train_loader, epochs=myEpoch)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_model(model, test_loader, f"{model_name}_{strategy}",fold, matchedSDS, res, wk)

# Test function
def test_model(model, test_loader, mname, fold, matchedSDS, res, wk):

    tags = mname.split('_')
    mod = tags[0].upper()
    var = tags[1] 

    model.eval()  # Set the model to evaluation mode
    all_predictions = []  # To store all high-resolution predictions
    all_test_hr = []  # To store all high-resolution inputs

    with torch.no_grad():
        for test_lr, test_hr in tqdm(test_loader, desc='Testing'):
            test_lr, test_hr  = test_lr.to(device), test_hr.to(device)

            # Process each sample in the batch
            batch_predictions = []
            for sample_ix in range(test_lr.size(0)):
                predicted_hr = model(test_lr[sample_ix].unsqueeze(0).unsqueeze(1))
                batch_predictions.append(predicted_hr[0, 0].cpu().numpy())
                all_predictions.append(predicted_hr.cpu().numpy())
                all_test_hr.append(test_hr[sample_ix].cpu().numpy())

    # Define your region of interest
    if res == '12km':
        xS, xE = [200, 800]
        yS, yE = [100, 400]

    # Compute differences
    hr_array = test_hr[sample_ix].cpu().numpy()
    difference_pred = hr_array - batch_predictions[sample_ix]
    difference_lr = hr_array - test_lr[sample_ix].cpu().numpy()  # unused in final plots but computed here if needed
    difference_ds = hr_array - matchedSDS
    difference_dspred = matchedSDS - batch_predictions[sample_ix]

    # Dictionaries for vmin/vmax
    vmin_dict = {
            'abs': {'t2':273,'ws':0,'vpd':50},
            'mdiff':{'t2':-2.5,'ws':-5,'vpd':-400},
        'diff':{'t2':-25,'ws':-15,'vpd':-400}}

    vmax_dict = {
            'abs': {'t2':'320','ws':'50','vpd':'2000'},
            'mdiff':{'t2':2.5,'ws':5,'vpd':400},
            'diff':{'t2':25,'ws':15,'vpd':400}}

    # Prepare data and plot configurations
    plots = [
        {
            'data': batch_predictions[sample_ix],
            'title': f'{mod} High Resolution',
            'cmap': 'viridis',
            'vmin': vmin_dict['abs'][var],
            'vmax': vmax_dict['abs'][var],
            'label': f'{varN[var]} [{units[var]}]'
        },
        {
            'data': hr_array,
            'title': 'True High Resolution',
            'cmap': 'viridis',
            'vmin': vmin_dict['abs'][var],
            'vmax': vmax_dict['abs'][var],
            'label': f'{varN[var]} [{units[var]}]'
        },
        {
            'data': matchedSDS,
            'title': 'S. Downscaled High Resolution',
            'cmap': 'viridis',
            'vmin': vmin_dict['abs'][var],
            'vmax': vmax_dict['abs'][var],
            'label': f'{varN[var]} [{units[var]}]'
        },
        {
            'data': difference_pred,
            'title': f'True High Res - {mod} High Res',
            'cmap': 'seismic',
            'vmin': vmin_dict['diff'][var],
            'vmax': vmax_dict['diff'][var],
            'label': f'Difference [{units[var]}]'
        },
        {
            'data': difference_dspred,
            'title': f'S. Downscaled - {mod} High Resolution',
            'cmap': 'seismic',
            'vmin': vmin_dict['mdiff'][var],
            'vmax': vmax_dict['mdiff'][var],
            'label': f'Difference [{units[var]}]'
        },
        {
            'data': difference_ds,
            'title': 'True High Res - S. Downscaled',
            'cmap': 'seismic',
            'vmin': vmin_dict['diff'][var],
            'vmax': vmax_dict['diff'][var],
            'label': f'Difference [{units[var]}]'
        },
    ]

    # Set up a figure with enough subplots (3x3, but we will use 6 of them)
    plt.figure(figsize=(15, 10))

    # Loop through each plot configuration
    for i, p in enumerate(plots, 1):
        plt.subplot(3, 3, i)
        im = plt.imshow(
            p['data'],
            cmap=p['cmap'],
            origin='lower',
            vmin=p.get('vmin', None),
            vmax=p.get('vmax', None)
        )

        if res == '12km':
            plt.xlim(xS, xE)
            plt.ylim(yS, yE)
        
        plt.title(p['title'])
        
        # Remove x and y ticks
        plt.xticks([])
        plt.yticks([])
        
        cbar = plt.colorbar(im, orientation='horizontal', pad=0.1, shrink=0.8)
        if 'label' in p:
            cbar.set_label(p['label'])

    plt.tight_layout()
    plt.savefig(f'./output/{res}/{wk}/{mname}_sample_{fold}.png', bbox_inches='tight')
    plt.close()

    return np.squeeze(np.array(all_predictions)), np.squeeze(np.array(all_test_hr))

# Function to evaluate model performance remains the same
def evaluate_model_performance(y_true, y_pred):
    """
    Calculate MAE, RMSE, and Pearson Correlation between actual and predicted values.
    
    Parameters:
    - y_true (array-like): Ground truth values.
    - y_pred (array-like): Predicted values from the model.
    
    Returns:
    - metrics (dict): Dictionary containing MAE, RMSE, and Pearson Correlation.
    """
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pear_corr, _ = pearsonr(y_true-np.mean(y_true), y_pred-np.mean(y_pred))

    sum_y = np.sum(y_true)
    sum_yhat = np.sum(y_pred)
    sum_y2 = np.sum(y_true ** 2)
    sum_yhat2 = np.sum(y_pred ** 2)
    sum_y_yhat = np.sum(y_true * y_pred)
    N = len(y_true)
    
    sum_squared_error = np.sum((y_true - y_pred) ** 2)
    sum_absolute_error = np.sum(np.abs(y_true - y_pred))

    #numerator = (N * sum_y_yhat) - (sum_y * sum_yhat)
    #denominator = np.sqrt((N * sum_y2 - sum_y**2) * (N * sum_yhat2 - sum_yhat**2))
    #pearson_corr = numerator / denominator
    #rmse = np.sqrt(sum_squared_error / N)
    #mae = sum_absolute_error / N

    return {"MAE": mae, "RMSE": rmse, "PearCorr": pear_corr, "N": N, 
            "sum_y":sum_y, "sum_y2":sum_y2, "sum_yhat":sum_yhat, "sum_yhat2":sum_yhat2, "sum_y_yhat":sum_y_yhat, 
            "SSE":sum_squared_error, "SAE":sum_squared_error }

def save_multiple_models(sampleNCF, model_results, res, wk):
    """
    Saves predictions and RMSE data from multiple models to a single NetCDF file.

    Parameters:
    sampleNCF : xarray DataArray or Dataset
        Reference file containing coordinate information.
    model_results : dict
        Dictionary where keys are model names and values are lists [prediction, truth].
        Example: {"unet_temperature_F": [unet_prediction, unet_truth], "cnn_temperature_F": [cnn_prediction, cnn_truth]}
    var : str
        Variable name (e.g., "temperature") to be saved in the NetCDF file.

    Returns:
    None
    """
    # Extract coordinates from the reference file
    time_coords = sampleNCF.time.values
    lat_coords = sampleNCF.latitude.values
    lon_coords = sampleNCF.longitude.values

    # Dictionary to store data variables
    data_vars = {}

    for model_name, modTrue in model_results.items():
        tags = model_name.split('_')
        modN = tags[0].upper()
        var = tags[1] 
        # Extract predictions and ground truth
        prediction = modTrue[0]  # Shape: (time, latitude, longitude)
        truth = modTrue[1]       # Shape: (time, latitude, longitude)

        # Compute RMSE across the time dimension
        rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=0))  # Shape: (latitude, longitude)

        # Add model-specific predicted values
        data_vars[f"{modN}_{var}_predicted"] = (["time", "latitude", "longitude"], prediction, {"units": units[var] })

        # Add RMSE for the model
        data_vars[f"{modN}_{var}_rmse"] = (["latitude", "longitude"], rmse, {"units": units[var] })

    # Create an xarray dataset
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": time_coords,
            "latitude": lat_coords,
            "longitude": lon_coords,
        },
        attrs={
            "description": f"Predictions and RMSE for multiple models ({', '.join(model_results.keys())})"
        }
    )

    # Sanitize filename (replace spaces and special characters)
    safe_var = re.sub(r'\W+', '_', var)
    filename = f"output/{res}/{wk}/DL_models_{safe_var}_predictions_with_rmse.nc"

    # Save to NetCDF file
    ds.to_netcdf(filename)

    print(f"NetCDF file saved successfully: {filename}")

##########

if __name__ == "__main__":

    #inR = 'CONUS404'
    inR = 'NAM'
    #wk = 'wk4'
    wk = 'wk1'

    if inR == 'CONUS404':
        res = '4km'
    elif inR == 'NAM':
        res = '12km'

    weekNum = {'wk1': 0,
    'wk2':7,
    'wk3':14,
    'wk4':21}

    for var in ['t2','ws','vpd']:
        if var == 'vpd':
            tave= 'avg' 
        else:
            tave= 'max' 

        df_sites = pd.read_csv('../raw_data/obs/SiteList_LatLon.csv')
        lat_values = df_sites['Latitude'].values
        lon_values = df_sites['Longitude'].values
        site_id = df_sites['Site'].values

        # Add SDS for comparing to HR and LR. See test model function
        file_list = glob.glob(f'../raw_data/UFS_SDS/{res}/{var}*{tave}/{var}*{tave.upper()}_UFS_S2S_FORECAST_Daily_*')
        dsSDS = open_files_with_subset(file_list,0+weekNum[wk],7+weekNum[wk])
        dsSDS = dsSDS.rename({old: new for old, new in {'t2m': 't2','wspeed':'ws'}.items() if old in dsSDS.data_vars})
        dsSDS = dsSDS.swap_dims({"valid_time": "time", "lon":"longitude", "lat":"latitude"})
        dsSDS = dsSDS.rename({"valid_time": "time", "lon":"longitude", "lat":"latitude"})
        dsSDS = dsSDS.set_index({ "time": "time", "longitude": "longitude", "latitude": "latitude"})
        dsSDS['longitude'] = (dsSDS.longitude + 180) % 360 - 180
        lat_indicesDS = [int(np.abs(dsSDS.latitude - lat).argmin()) for lat in lat_values]
        lon_indicesDS = [int(np.abs(dsSDS.longitude - lon).argmin()) for lon in lon_values]

        file_list = glob.glob(f'../raw_data/UFS_S2S/{var}_daily_{tave}/{var}_UFS_S2S_FORECAST_{tave.upper()}_Daily_*')
        dsL = open_files_with_subset(file_list,0+weekNum[wk],7+weekNum[wk])

        file_list = glob.glob(f'../raw_data/{inR}/{var}_daily_{tave}/{var}_{inR}_*_{tave.upper()}_Daily_*')
        dsH = xr.open_mfdataset(file_list,combine="by_coords").chunk({"time": 10})

        low_res_data, high_res_data = regrid(dsL,dsH,var)
        lat_indices = [int(np.abs(high_res_data.latitude - lat).argmin()) for lat in lat_values]
        lon_indices = [int(np.abs(high_res_data.longitude - lon).argmin()) for lon in lon_values]

        # Dictionary to store models and results
        results = {}
        # Initialize an empty list to store results for the DataFrame
        results_list = []

        random_seed = 7
        n_splits = 1
        data_splits = split_data_for_training_and_testing(low_res_data, high_res_data,random_seed=random_seed)
        folds =  [data_splits] 

        #Get Test dates to evaluate at monitors 
        selectedH = data_splits['test_high_res'][var].isel(latitude=xr.DataArray(lat_indices, dims="points"),
            longitude=xr.DataArray(lon_indices, dims="points")).sel(time=data_splits['test_high_res'].time)
        dfH = selectedH.to_dataframe().reset_index().rename(columns={var:'HR'})
        dfL = data_splits['test_low_res'][var].isel(latitude=xr.DataArray(lat_indices, dims="points"),
            longitude=xr.DataArray(lon_indices, dims="points")).sel(time=data_splits['test_high_res'].time).to_dataframe().reset_index().rename(columns={var:'LR'})
        df_combined = pd.merge(dfH, dfL, on=['time', 'points', 'longitude', 'latitude'], how='outer')

        dsSDS_matched = dsSDS.sel(latitude=dsH.latitude, longitude=dsH.longitude,time=selectedH.time[-1], method='nearest')[var].values
        
        # Loop over folds and prepare data
        for fold_idx, data_splits in enumerate(folds):
            print(f"Processing Fold {fold_idx + 1}/{n_splits}...")

            torch.cuda.empty_cache()
            gc.collect()
            
            # Loop through each key in data_splits and replace NaN with 288 (mean)
            train_dataset, test_dataset = prepare_data(data_splits, lr_var=var, hr_var=var)

            # Instantiate, train, and test each model
            # U-Net Model
            unet_model = UNetDownscale(input_channels=1, predictor_channels=None, upscale_factor=1)
            unetF = train_and_test_model(unet_model, 'F', train_dataset, test_dataset, f'unet_{var}', fold_idx, dsSDS_matched, res, wk)
            
            # CNN Model
            cnn_model = CNNDownscale(input_channels=1, predictor_channels=None, upscale_factor=1)
            cnnF = train_and_test_model(cnn_model, 'F', train_dataset, test_dataset, f'cnn_{var}', fold_idx, dsSDS_matched, res, wk)

            # SRGAN Generator Model
            srgan_generator = SRGANGenerator(input_channels=1, predictor_channels=None, upscale_factor=1)
            sgnF = train_and_test_model(srgan_generator, 'F', train_dataset, test_dataset, f'srgan_{var}', fold_idx, dsSDS_matched, res, wk)

            # Example model results dictionary with only the 0 and mean strategies
            model_results = {f"unet_{var}_F": unetF, f"cnn_{var}_F": cnnF, f"srgan_{var}_F": sgnF}

            save_multiple_models(data_splits['test_high_res'][var].sel(time=data_splits['test_high_res'].time), model_results, res, wk)

