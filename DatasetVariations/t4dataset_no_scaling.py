from torch.utils.data import Dataset,DataLoader
import xarray as xr
import torch
import numpy as np
import pandas as pd

class T4DatasetNoScaling(Dataset):
    """
    Pytorch Dataset for mapping timesteps of IMERG and gencast for T-4 Timesteps forecast of each event.
    """
    def __init__(self,mapping,data,step=25):
        """
        Args:
            mapping(dict): Dictionary mapping of IMERG to prediction file paths.
        """
        self.file_mapping=mapping
        self.index=[]
        self.data=data

        if data=='forecast':
            for imerg_file, pred_file in self.file_mapping.items():
                self._process_file_pair(imerg_file, pred_file,step=step)
            print(f"Dataset created with {len(self.index)} pairs of adjacent timesteps for IMERG and GenCast")
        if data=='era5':
            for imerg_file,era_file in self.file_mapping.items():
                self._process_file_pair(imerg_file,era_file,step=0)
            print(f"Dataset created with {len(self.index)} pairs of adjacent timesteps for IMERG and ERA5")

    def _process_file_pair(self,imerg_file_path,pred_file_path,step):
        """Process and create index of imerg to corresponding prediction timestep."""
        try:
            with xr.open_dataset(imerg_file_path,decode_timedelta=True) as imerg_data, xr.open_dataset(pred_file_path,decode_timedelta=True) as pred_data:
                # imerg_data=xr.open_dataset(imerg_file_path)
                # pred_data=xr.open_dataset(pred_file_path)

                pred_times=pred_data.time.values
                if 'sample' in pred_data.coords:
                    for num_sample in range(pred_data.sample.size):
                        for i in range(len(pred_times)-1):
                            self.index.append({
                                'imerg_file': imerg_file_path,
                                'pred_file': pred_file_path,
                                'sample_index':num_sample,
                                'current_pred_idx': i,
                                'next_pred_idx': i + 1,
                                # for imerg step would be 25
                                'current_imerg_idx': step + i,
                                'next_imerg_idx': step + i + 1 ,
                            })
                else:
                    for i in range(len(pred_times) - 1 - 1):
                        self.index.append({
                            'imerg_file': imerg_file_path,
                            'pred_file': pred_file_path,
                            'sample_index': None,
                            'current_pred_idx': i + 1,
                            'next_pred_idx': i + 2,
                            # for imerg step would be 25
                            'current_imerg_idx': step + i,
                            'next_imerg_idx': step + i + 1 ,
                        })
        except Exception as e:
            print(f"Error processing files {imerg_file_path} and {pred_file_path}: {e}") 
            
    def __len__(self):
        """Return the size of the index i.e. the number of pieces of the datasets"""
        return len(self.index)

    def __getitem__(self,idx):
        """Returns the input predictions and corresponding IMERG data to be mapped to.

        Returns:
            torch.Tensor: Prediction data tensor
            torch.Tensor: IMERG data tensor
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_info = self.index[idx]
        
        # Load the data from files
        with xr.open_dataset(sample_info['imerg_file'],decode_timedelta=True) as imerg_data,xr.open_dataset(sample_info['pred_file'],decode_timedelta=True) as pred_data:
        
            if 'sample' in pred_data.coords:
                pred = pred_data.isel(time=slice(sample_info['current_pred_idx'],sample_info['next_pred_idx']+1),batch=0,sample=sample_info['sample_index']).sel(lat=slice(5,40), lon=slice(60,100))*1000
            else :
                pred = pred_data.isel(time=slice(sample_info['current_pred_idx'],sample_info['next_pred_idx']+1),batch=0).sel(lat=slice(5,40), lon=slice(60,100))*1000

            imerg = imerg_data.isel(time=slice(sample_info['current_imerg_idx'],sample_info['next_imerg_idx']+1)).transpose('time','lat','lon').sel(lat=slice(5,40), lon=slice(60,100))

            pred_data_tensor = torch.tensor(pred['total_precipitation_12hr'].values.astype(np.float32)).unsqueeze(1)
            imerg_data_tensor = torch.tensor(imerg.precipitation.values.astype(np.float32)).unsqueeze(1)

            return pred_data_tensor,imerg_data_tensor