from torch.utils.data import Dataset,DataLoader
import xarray as xr
import torch
import numpy as np
import pandas as pd

class TestDataset(Dataset):
    """
    Class for creating the test dataset for the model
    """

    # TODO: Thinking of making it over necessary data 
    # Like it can take a step as input which would bring it over event dates for extremes
    # Or we can just test over the entire dataset

    def __init__(self,mapping,step_imerg,step_pred,window_size,scaling_values) -> None:
        """
        Args:
            mapping(dict): Dictionary mapping IMERG to Predictions
            step_imerg(int): Number of steps Imerg has to be shifted to map it to gencast on the same timestamp. 25 will work to bring it around extreme event dates
            step_pred(int): Number of steps gen predictions time axis is to be shifted to bring it around and over extreme event.
            window_size: Number of timesteps to be considered in a slice
            scaling_type: Type of scaling "global_min_max_norm","grid_wise_min_max" etc.
            scaling_values: A particular value in case of global_min_max or multiple values when we have the case of grid wise normalization.
        """

        super().__init__()
        self.file_mapping=mapping
        self.index=[]
        # TODO: When adding code for grid_wise_min_max one great idea would be to use len of scaling values to find if it is an array or not and then normalize accordingly
        self.scaling_values=scaling_values
        self.epsilon = 1e-8  # to prevent division by zero
        self.step_imerg=step_imerg
        self.step_pred=step_pred

        for imerg_file, pred_file in self.file_mapping.items():
            self._process_file_pair(imerg_file, pred_file,window_size)
        print(f"Dataset created with {len(self.index)} pairs of adjacent timesteps for IMERG and GenCast")


    def _process_file_pair(self,imerg_file_path,pred_file_path,window_size):
        """Process and create index of imerg to corresponding prediction timestep."""
        try:
            with xr.open_dataset(imerg_file_path,decode_timedelta=True) as imerg_data, xr.open_dataset(pred_file_path,decode_timedelta=True) as pred_data:
                # imerg_data=xr.open_dataset(imerg_file_path)
                # pred_data=xr.open_dataset(pred_file_path)

                pred_times=pred_data.time.values
                if 'sample' in pred_data.coords:
                    for num_sample in range(pred_data.sample.size):
                        for i in range(len(pred_times)-window_size+1):
                            self.index.append({
                                'imerg_file': imerg_file_path,
                                'pred_file': pred_file_path,
                                'sample_index':num_sample,
                                # WARNING: Check whether this self.step_pred gets properly aligned or not
                                'current_pred_idx': i + self.step_pred,
                                # next_pred_idx is basically the end of the window
                                'next_pred_idx': i + window_size - 1 + self.step_pred,
                                # for imerg step would be 25 when i was working with T-4 days that is 8 timesteps in order to bring it from 11 to 25 in case of july_2005 
                                'current_imerg_idx': self.step_imerg + i,
                                'next_imerg_idx': self.step_imerg + window_size + i - 1,
                                'date' : imerg_data.time.values[self.step_imerg+i]
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

            pred_tensor = torch.tensor(pred['total_precipitation_12hr'].values.astype(np.float32)).unsqueeze(1)
            imerg_tensor = torch.tensor(imerg.precipitation.values.astype(np.float32)).unsqueeze(1)

            #NOTE: When doing cell wise normalization here just dividing by scaling values wont work.
            if self.scaling_values and self.scaling_values > 0:
                pred_tensor = pred_tensor / (self.scaling_values + self.epsilon)
                imerg_tensor = imerg_tensor / (self.scaling_values + self.epsilon)

            return pred_tensor, imerg_tensor
