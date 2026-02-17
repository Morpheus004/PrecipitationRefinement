from torch.utils.data import Dataset
import torch
import xarray as xr
import glob
import os
from cust_types import ScalingType
import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class JJASDatasetMean(Dataset):
    def __init__(self,pred_dir_path,imerg_path,window_size,scaling_type:ScalingType, climatology_path='/scratch/IITB/monsoon_lab/24d1236/pratham/Datasets/IMERG_consolidated/IMERG_JJAS_climatology_final_stacked_1degree.nc', global_max = None):
        """
        Args:
            pred_dir(str): Directory where all predictions are present
            imerg_path(str): Path of the imerg_variable
            window_size(int) : Number of timesteps to pick from the predictions.
            scaling_type(str): Set scaling to GlobalMax, NoScaling, BoxCox, or GridwiseAnomaly
            climatology_path(str): Path to climatology file (required for GridwiseAnomaly)
        """
        logger.info("Start data set processing")
        logger.debug(f"Pred:{pred_dir_path}")
        logger.debug(f"IMERG:{imerg_path}")
        self.window_size=window_size
        self.pred_files = sorted(glob.glob(f"{pred_dir_path}/*"),key=lambda x:int(os.path.basename(x).split('_')[0]))
        self.pred = xr.open_mfdataset(self.pred_files,decode_timedelta=True,preprocess=self._preprocess).mean('sample').compute()
        self.imerg = xr.open_dataset(imerg_path)
        self.scaling_type = scaling_type
        if scaling_type == 'GlobalMax':
            if global_max is None:
                logger.info(f"ScalingType is {scaling_type}")
                self.global_max = self.pred['total_precipitation_12hr'].max().values*1000
            else: 
                self.global_max = global_max
        elif scaling_type == 'BoxCox':
            logger.info(f"ScalingType is {scaling_type}")
            flat_precip = self.pred['total_precipitation_12hr'].sel(lat=slice(5,40), lon=slice(60,100)).values.flatten() * 1000
            flat_precip_changed = flat_precip[flat_precip > 0]
            # Compute Box-Cox transform and store lambda
            _, self.boxcox_lambda = stats.boxcox(flat_precip_changed)
        elif scaling_type == "GridwiseAnomaly":
            logger.info(f"ScalingType is {scaling_type}")
            # if climatology_path:
            #     # Load climatology data
            #     self.climatology = xr.open_dataset(climatology_path)
            #     # Create a mapping from (day, hour) to climatology index
            #     self.clim_mapping = {}
            #     for i, (day, hour) in enumerate(zip(self.climatology.dayofyear.values, self.climatology.hour.values)):
            #         self.clim_mapping[(day, hour)] = i
            # else:
                # raise ValueError("climatology_path is required for GridwiseAnomaly scaling")
            if climatology_path:
                # Load climatology data
                self.climatology = xr.open_dataset(climatology_path)
                
                # Ensure dayofyear and hour are arrays aligned to 'time'
                dayofyear_arr = self.climatology['dayofyear'].values
                hour_arr = self.climatology['hour'].values
                
                # Create a pandas MultiIndex from (dayofyear, hour)
                multi_index = pd.MultiIndex.from_arrays([dayofyear_arr, hour_arr], names=('dayofyear', 'hour'))
                
                # Map (dayofyear, hour) → time index position
                self.clim_mapping = {key: idx for idx, key in enumerate(multi_index)}

            else:
                raise ValueError("climatology_path is required for GridwiseAnomaly scaling")

    @staticmethod
    def _preprocess(df):
        return df['total_precipitation_12hr']
            
    def __len__(self):
        batch,time,_,_=self.pred.sizes.values()
        return batch

    def __getitem__(self,idx):
        first_timestamp=self.pred.isel(batch=int(idx),time=0).datetime.values
        last_timestamp=self.pred.isel(batch=int(idx),time=self.window_size-1).datetime.values
        
        pred_tensor=torch.tensor(self.pred.isel(batch=int(idx),time=slice(0,self.window_size)).sel(lat=slice(5,40), lon=slice(60,100))['total_precipitation_12hr'].values*1000)
        imerg_tensor=torch.tensor(self.imerg.sel(time=slice(first_timestamp,last_timestamp)).sel(lat=slice(5,40), lon=slice(60,100))['precipitation'].transpose('time','lat','lon').values)

        pred_tensor[pred_tensor < 0] = 0
        pred_tensor=pred_tensor.unsqueeze(1)
        imerg_tensor=imerg_tensor.unsqueeze(1)

        if self.scaling_type == 'GlobalMax':
            return pred_tensor / self.global_max, imerg_tensor / self.global_max
        elif self.scaling_type == 'BoxCox':
            epsilon = 1e-6
            pred_tensor[pred_tensor <= 0] = 0
            pred_tensor_bc = torch.tensor(stats.boxcox((pred_tensor.numpy() + epsilon).flatten(), lmbda=self.boxcox_lambda),dtype=torch.float32).reshape(pred_tensor.shape)
            imerg_tensor_bc = torch.tensor(stats.boxcox((imerg_tensor.numpy() + epsilon).flatten(), lmbda=self.boxcox_lambda),dtype=torch.float32).reshape(imerg_tensor.shape)
            return pred_tensor_bc, imerg_tensor_bc
        elif self.scaling_type == 'GridwiseAnomaly':
            # Calculate anomalies for both pred and imerg tensors
            pred_anomalies = self._calculate_anomalies(pred_tensor, first_timestamp, last_timestamp)
            imerg_anomalies = self._calculate_anomalies(imerg_tensor, first_timestamp, last_timestamp)
            return pred_anomalies, imerg_anomalies
        else:
            return pred_tensor, imerg_tensor

    def _calculate_anomalies(self, tensor, first_timestamp, last_timestamp):
        """
        Calculate anomalies by subtracting climatology values for each lat/lon cell
        """
        # Get the time range for this sample
        time_range = self.imerg.sel(time=slice(first_timestamp, last_timestamp)).time
        
        anomalies = torch.zeros_like(tensor)
        
        for t_idx, timestamp in enumerate(time_range):
            # Extract day and hour from timestamp
            dt = timestamp.dt
            day = int(dt.dayofyear.values)
            hour = int(dt.hour.values)
            
            # Get climatology index
            clim_idx = self.clim_mapping.get((day, hour))
            if clim_idx is not None:
                # Get climatology values for this day/hour
                clim_values = self.climatology.isel(time=clim_idx).sel(lat=slice(5,40), lon=slice(60,100))['precipitation_mean'].values
                
                # Calculate anomaly: value - climatology
                # Note: tensor shape is (time, channel, lat, lon) where channel=1
                anomalies[t_idx, 0, :, :] = tensor[t_idx, 0, :, :] - torch.tensor(clim_values)
            else:
                logger.error(f"No matching entry for this particular datetime in climatology :{str(day)} and {str(hour)}")
                raise RuntimeError("Climatology data created does not have the necessary dates")
        
        return anomalies

    
