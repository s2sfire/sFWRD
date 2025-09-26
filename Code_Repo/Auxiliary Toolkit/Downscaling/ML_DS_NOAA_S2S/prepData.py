import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import pandas as pd
import numpy as np
import sys
import xesmf as xe
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import KFold

def plot_and_save_comparison(dsLR_rgrd, dsHR, variable='t2', time_idx=0, output_file='comparison_plot.png'):
    """
    Plots two datasets side by side and saves the plot to a file.

    Parameters:
        dsLR_rgrd (xarray.Dataset): Regridded low-resolution dataset.
        dsHR (xarray.Dataset): High-resolution dataset.
        variable (str): Name of the variable to plot.
        time_idx (int): Time index to select for plotting.
        output_file (str): Path to save the output plot image.
    """
    # Select the specified variable and time slice
    data_lr = dsLR_rgrd[variable].isel(time=time_idx)
    data_hr = dsHR[variable].isel(time=time_idx)

    # Create the plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Plot low-resolution regridded dataset
    data_lr.plot(ax=axes[0], cmap='viridis')
    axes[0].set_title(f"Low-Resolution Regridded ({variable})")

    # Plot high-resolution dataset
    data_hr.plot(ax=axes[1], cmap='viridis')
    axes[1].set_title(f"High-Resolution ({variable})")

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Comparison plot saved to {output_file}")

# Define a preprocessing function with slice size as an argument
def subset_valid_time(ds, ss, se):
    return ds.isel(valid_time=slice(ss, se))

# Wrapper function for xarray's open_mfdataset
def open_files_with_subset(file_list, ss, se):
    return xr.open_mfdataset(
        file_list,
        combine="by_coords",
        preprocess=lambda ds: subset_valid_time(ds, ss=ss, se=se),
        chunks={"valid_time": se-ss}
    )

def regrid(dsLRo,dsHRo,var='t2'):
    #dsLRo = xr.open_dataset('../data/UFS_MPM/t2m_UFS_S2S_FORECAST_AVG_lead01.nc')
    #dsHRo = xr.open_dataset('../data/NAM/t2_NAM_HISTORICAL_AVG_Daily_2011.nc')

    # Step 1: Find the intersection of the time steps between dsLRo and dsHRo
    common_times = np.intersect1d(dsLRo.valid_time.values, dsHRo.time.values)

    # Step 2: Subset both datasets to the common time steps
    dsLR = dsLRo.sel(valid_time=common_times)
    dsHR = dsHRo.sel(time=common_times)

    
    # Step 3: Swap valid_time with time in dsLR to avoid renaming conflicts
    dsLR = dsLR.swap_dims({"valid_time": "time"})

    # Step 4: Rename variables to match (assuming 't2m' in dsLR and 't2' in dsHR)
    #dsLR = dsLR.rename({'t2m': 't2'})

    # Step 5: Define the output grid based on the high-resolution dataset's latitude and longitude
    ds_out = xr.Dataset(
        {
            "latitude": dsHR.latitude,
            "longitude": dsHR.longitude,
        }
    )

    # Step 6: Create a regridder using xESMF
    regridder = xe.Regridder(dsLR, ds_out, 'nearest_s2d')

    # Step 7: Apply the regridding to the low-resolution dataset
    dsLR_rgrd = regridder(dsLR)

    # Step 8: Keep only the first value along the lead_week dimension and drop extra coordinates
    #dsLR_rgrd = dsLR_rgrd.isel(lead_week=0).drop_vars(['lead_week', 'heightAboveGround', 'valid_time'], errors='ignore')
    dsLR_rgrd = dsLR_rgrd.drop_vars(['step', 'valid_time'], errors='ignore')
    
    # Step 9: Align dsLR_rgrd time with dsHR time to ensure exact matching
    dsLR_rgrd['time'] = dsHR['time']

    # Step 10: Apply NaN mask from dsHR to dsLR_rgrd
    #nan_mask_hr = np.isnan(dsHR[var])
    #dsLR_rgrd = dsLR_rgrd.where(~nan_mask_hr, other=np.nan)
    
    # Step 10: Apply Low Res to High Res NaNs
    dsHR[var] = dsHR[var].fillna(dsLR_rgrd[var])


    return [dsLR_rgrd, dsHR]

def split_data_for_training_and_testing(low_res_data, high_res_data, low_res_time_var='time', high_res_time_var='time', train_ratio=0.8, random_split=True, random_seed=None):
    """
    Splits high-resolution and low-resolution NetCDF data into training and testing sets, ensuring 
    equal representation from each season, assuming exact date matches between datasets.

    Parameters:
    low_res_data (xarray.Dataset): Low-resolution dataset.
    high_res_data (xarray.Dataset): High-resolution dataset.
    low_res_time_var (str): Name of the time variable in the low-resolution file.
    high_res_time_var (str): Name of the time variable in the high-resolution file.
    train_ratio (float): Proportion of data to use for training (e.g., 0.8 for 80% training).
    random_split (bool): Whether to split randomly within each season.

    Returns:
    dict: A dictionary with training and testing sets for both high and low-resolution data.
    """
    # Define a function to determine the season based on the month
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'

    # Add season as a coordinate for both datasets
    high_res_data['season'] = xr.DataArray(
        [get_season(pd.Timestamp(time).month) for time in high_res_data[high_res_time_var].values],
        dims=high_res_time_var
    )
    low_res_data['season'] = xr.DataArray(
        [get_season(pd.Timestamp(time).month) for time in low_res_data[low_res_time_var].values],
        dims=low_res_time_var
    )

    # Pr:epare lists to collect training and testing datasets
    train_data_high_list = []
    test_data_high_list = []
    train_data_low_list = []
    test_data_low_list = []
    
    if random_seed is not None:
        np.random.seed(random_seed)

    # Iterate over each season and split data within each season
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        # Filter data by season
        season_data_high = high_res_data.where(high_res_data['season'] == season, drop=True)
        season_data_low = low_res_data.where(low_res_data['season'] == season, drop=True)

        # Check that time steps match between datasets
        if len(season_data_high[high_res_time_var]) != len(season_data_low[low_res_time_var]):
            raise ValueError(f"Time dimensions do not match for season: {season}")

        # Calculate the number of samples for training and testing within each season
        season_time_steps = len(season_data_high[high_res_time_var])
        train_size = int(season_time_steps * train_ratio)

        # Generate indices for splitting
        indices = np.arange(season_time_steps)
        if random_split:
            np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Split high-resolution and low-resolution data for the season
        train_data_high_list.append(season_data_high.isel({high_res_time_var: train_indices}))
        test_data_high_list.append(season_data_high.isel({high_res_time_var: test_indices}))

        train_data_low_list.append(season_data_low.isel({low_res_time_var: train_indices}))
        test_data_low_list.append(season_data_low.isel({low_res_time_var: test_indices}))

    # Concatenate seasonal data to form the complete training and testing datasets
    train_data_high = xr.concat(train_data_high_list, dim=high_res_time_var)
    test_data_high = xr.concat(test_data_high_list, dim=high_res_time_var)

    train_data_low = xr.concat(train_data_low_list, dim=low_res_time_var)
    test_data_low = xr.concat(test_data_low_list, dim=low_res_time_var)
    
    if ~np.any(test_data_low.time.values == test_data_high.time.values):
        sys.exit('Error:Test dates do not match between low and high Res') 

    return {
        'train_high_res': train_data_high,
        'test_high_res': test_data_high,
        'train_low_res': train_data_low,
        'test_low_res': test_data_low
    }

def split_data_for_kfold_cross_validation(low_res_data, high_res_data, low_res_time_var='time', high_res_time_var='time', n_splits=5, random_seed=None):
    """
    Splits high-resolution and low-resolution NetCDF data into folds for k-fold cross-validation.

    Parameters:
    low_res_data (xarray.Dataset): Low-resolution dataset.
    high_res_data (xarray.Dataset): High-resolution dataset.
    low_res_time_var (str): Name of the time variable in the low-resolution file.
    high_res_time_var (str): Name of the time variable in the high-resolution file.
    n_splits (int): Number of folds for k-fold cross-validation.
    random_seed (int): Random seed for reproducibility.

    Returns:
    list of dict: A list of dictionaries for each fold, containing training and testing sets for both datasets.
    """
    # Ensure datasets have the same time steps
    if not np.array_equal(low_res_data[low_res_time_var].values, high_res_data[high_res_time_var].values):
        raise ValueError("Time dimensions do not match between low and high-resolution datasets.")
    
    # Get time indices and set up KFold
    time_indices = np.arange(len(low_res_data[low_res_time_var]))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # Store splits for each fold
    folds = []
    
    for train_indices, test_indices in kf.split(time_indices):
        # Select train/test data for low-resolution dataset
        train_low_res = low_res_data.isel({low_res_time_var: train_indices})
        test_low_res = low_res_data.isel({low_res_time_var: test_indices})
        
        # Select train/test data for high-resolution dataset
        train_high_res = high_res_data.isel({high_res_time_var: train_indices})
        test_high_res = high_res_data.isel({high_res_time_var: test_indices})
        
        # Append the split to the folds list
        folds.append({
            'train_high_res': train_high_res,
            'test_high_res': test_high_res,
            'train_low_res': train_low_res,
            'test_low_res': test_low_res
        })
    
    return folds

def prepare_data(data_splits, lr_var='t2', hr_var='t2', batch_size=16, pin_memory=False):
    
    # Access the training and testing sets
    train_hr_data = torch.tensor(data_splits['train_high_res'][hr_var].values, dtype=torch.float32)
    test_hr_data  = torch.tensor(data_splits['test_high_res'][hr_var].values, dtype=torch.float32)
    train_lr_data = torch.tensor(data_splits['train_low_res'][lr_var].values, dtype=torch.float32)
    test_lr_data  = torch.tensor(data_splits['test_low_res'][lr_var].values, dtype=torch.float32)

    train_dataset = TensorDataset(train_lr_data, train_hr_data)
    test_dataset = TensorDataset(test_lr_data, test_hr_data)
    
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    #return train_loader, test_loader
    return train_dataset, test_dataset

if __name__ == "__main__":
    # Run the function to regrid, remove common NaNs, and return the datasets

    if False:
        file_list = glob.glob('../raw_data/UFS_S2S/t2_daily_max/t2_UFS_S2S_FORECAST_MAX_Daily_201*')
        dsL = open_files_with_subset(file_list,0,7)
        file_list = glob.glob('../raw_data/NAM/t2_daily_max/t2_NAM_HISTORICAL_MAX_Daily_20*')
        dsH = xr.open_mfdataset(file_list,combine="by_coords").chunk({"time": 10})
        
        dsLR, dsHR = regrid(dsL,dsH,'t2')
        
        data_splits = split_data_for_training_and_testing(dsLR, dsHR) 
        # Loop through each key in data_splits and replace NaN with 0
        data_splits_filled = {key: data_splits[key].fillna(0) for key in data_splits}
        train_loader, test_loader = prepare_data(data_splits_filled)
        # Example usage:
        #plot_and_save_comparison(dsLR, dsHR, variable='t2', time_idx=0, output_file='comparison_plot.png')
