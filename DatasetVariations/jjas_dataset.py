from torch.utils.data import Dataset
import torch
import xarray as xr
import glob
import os
from cust_types import ScalingType
import numpy as np
from scipy import stats

class JJASDataset(Dataset):
    def __init__(self,pred_dir_path,imerg_path,window_size,scaling_type:ScalingType):
        """
        Args:
            pred_dir(str): Directory where all predictions are present
            imerg_path(str): Path of the imerg_variable
            window_size(int) : Number of timesteps to pick from the predictions.
            scaling_type(str): Set scaling to GlobalMax or NoScaling
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
        else:
            return pred_tensor, imerg_tensor

    
