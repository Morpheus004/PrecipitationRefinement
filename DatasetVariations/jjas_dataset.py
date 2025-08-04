from torch.utils.data import Dataset
import torch
import xarray as xr
import glob
import os
from cust_types import ScalingType
import numpy as np
from scipy import stats

class JJASDataset(Dataset):
    def __init__(self,pred_dir_path,imerg_path,window_size,scaling_type:ScalingType, climatology_path="/scratch/IITB/monsoon_lab/24d1236/pratham/Datasets/jjas_clim_imerg/IMERG_JJAS_climatology_final_1_deg.nc"):
        """
        Args:
            pred_dir(str): Directory where all predictions are present
            imerg_path(str): Path of the imerg_variable
            window_size(int) : Number of timesteps to pick from the predictions.
            scaling_type(str): Set scaling to GlobalMax, NoScaling, BoxCox, or GridwiseAnomaly
            climatology_path(str): Path to climatology file (required for GridwiseAnomaly)
        """
        self.window_size=window_size
        self.pred_files = sorted(glob.glob(f"{pred_dir_path}/*"),key=lambda x:int(os.path.basename(x).split('_')[0]))
        self.pred = xr.open_mfdataset(self.pred_files,decode_timedelta=True,preprocess=self._preprocess)
        self.imerg = xr.open_dataset(imerg_path)
        self.scaling_type = scaling_type
        if scaling_type == 'GlobalMax':
            self.global_max = self.pred['total_precipitation_12hr'].max().values*1000
        elif scaling_type == 'BoxCox':
            flat_precip = self.pred['total_precipitation_12hr'].values.flatten() * 1000
            flat_precip_changed = flat_precip[flat_precip > 0]
            # Compute Box-Cox transform and store lambda
            _, self.boxcox_lambda = stats.boxcox(flat_precip_changed)
        elif scaling_type == "GridwiseAnomaly":
            if climatology_path:
                # Load climatology data
                self.climatology = xr.open_dataset(climatology_path)
                # Create a mapping from (day, hour) to climatology index
                self.clim_mapping = {}
                for i, (day, hour) in enumerate(zip(self.climatology.day.values, self.climatology.hour.values)):
                    self.clim_mapping[(day, hour)] = i
            else:
                raise ValueError("climatology_path is required for GridwiseAnomaly scaling")
            
    @staticmethod
    def _preprocess(df):
        return df['total_precipitation_12hr']
            
    def __len__(self):
        sample,batch,time,_,_=self.pred.sizes.values()
        return sample*batch

    def __getitem__(self,idx):
        sample_size = self.pred.sizes['sample']
        first_timestamp=self.pred.isel(sample=int(idx%sample_size),batch=int(idx//sample_size),time=0).datetime.values
        last_timestamp=self.pred.isel(sample=int(idx%sample_size),batch=int(idx//sample_size),time=self.window_size-1).datetime.values
        
        pred_tensor=torch.tensor(self.pred.isel(sample=int(idx%8),batch=int(idx//8),time=slice(0,self.window_size)).sel(lat=slice(5,40), lon=slice(60,100))['total_precipitation_12hr'].values*1000)
        imerg_tensor=torch.tensor(self.imerg.sel(time=slice(first_timestamp,last_timestamp)).sel(lat=slice(5,40), lon=slice(60,100))['precipitation'].transpose('time','lat','lon').values)

        pred_tensor=pred_tensor.unsqueeze(1)
        imerg_tensor=imerg_tensor.unsqueeze(1)

        if self.scaling_type == 'GlobalMax':
            return pred_tensor / self.global_max, imerg_tensor / self.global_max
        elif self.scaling_type == 'BoxCox':
            epsilon = 1e-6
            pred_tensor_bc = torch.tensor(stats.boxcox((pred_tensor.numpy() + epsilon).flatten(), lmbda=self.boxcox_lambda)).reshape(pred_tensor.shape)
            imerg_tensor_bc = torch.tensor(stats.boxcox((imerg_tensor.numpy() + epsilon).flatten(), lmbda=self.boxcox_lambda)).reshape(imerg_tensor.shape)
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
            day = dt.dayofyear.values
            hour = dt.hour.values
            
            # Get climatology index
            clim_idx = self.clim_mapping.get((day, hour))
            if clim_idx is not None:
                # Get climatology values for this day/hour
                clim_values = self.climatology.isel(time=clim_idx).sel(lat=slice(5,40), lon=slice(60,100))['precipitation'].values
                
                # Calculate anomaly: value - climatology
                # Note: tensor shape is (time, channel, lat, lon) where channel=1
                anomalies[t_idx, 0, :, :] = tensor[t_idx, 0, :, :] - torch.tensor(clim_values)
        
        return anomalies

    
