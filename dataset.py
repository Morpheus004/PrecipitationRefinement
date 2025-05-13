from torch.utils.data import Dataset,DataLoader
import xarray as xr
import torch
import numpy as np
import pandas as pd

class T4Dataset(Dataset):
    """
    Pytorch Dataset for mapping timesteps of IMERG and gencast for T-4 Timesteps forecast of each event.
    """
    def __init__(self,mapping):
        """
        Args:
            mapping(dict): Dictionary mapping of IMERG to prediction file paths.
        """
        self.file_mapping=mapping
        self.index=[]

        for imerg_file, pred_file in self.file_mapping.items():
            self._process_file_pair(imerg_file, pred_file)
        print(f"Dataset created with {len(self.index)} pairs of adjacent timesteps")

    def _process_file_pair(self,imerg_file_path,pred_file_path):
        """Process and create index of imerg to corresponding prediction timestep."""
        try:
            imerg_data=xr.open_dataset(imerg_file_path)
            pred_data=xr.open_dataset(pred_file_path)

            pred_times=pred_data.time.values

            for num_sample in range(pred_data.sample.size):
                for i in range(len(pred_times)-1):
                    self.index.append({
                        'imerg_file': imerg_file_path,
                        'pred_file': pred_file_path,
                        'sample_index':num_sample,
                        'current_pred_idx': i,
                        'next_pred_idx': i + 1,
                        'current_imerg_idx': 25 + i,
                        'next_imerg_idx': 25 + i + 1 ,
                    })
            # Close the datasets
            imerg_data.close()
            pred_data.close()
            
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
        with xr.open_dataset(sample_info['imerg_file']) as imerg_data,xr.open_dataset(sample_info['pred_file']) as pred_data:
        
            pred = pred_data.isel(time=slice(sample_info['current_pred_idx'],sample_info['next_pred_idx']+1),batch=0,sample=sample_info['sample_index']).sel(lat=slice(5,40), lon=slice(60,100))
            imerg = imerg_data.isel(time=slice(sample_info['current_imerg_idx'],sample_info['next_imerg_idx']+1)).transpose('time','lat','lon').sel(lat=slice(5,40), lon=slice(60,100))

            pred_data_tensor = torch.tensor(pred['total_precipitation_12hr'].values.astype(np.float32))
            imerg_data_tensor = torch.tensor(imerg.precipitation.values.astype(np.float32))

            return pred_data_tensor,imerg_data_tensor