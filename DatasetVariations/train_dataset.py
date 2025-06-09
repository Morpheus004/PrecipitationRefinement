from torch.utils.data import Dataset
import xarray as xr
import torch
import numpy as np
import sys
from tqdm import tqdm

class TrainDataset(Dataset):
    """
    Class for creating the training dataset for the model with support for different normalization schemes.
    """

    def __init__(self, mapping, step_imerg, step_pred, window_size, scaling_type="none", scaling_values=None, max_calc_source="both") -> None:
        """
        Args:
            mapping(dict): Dictionary mapping IMERG to Predictions
            step_imerg(int): Number of steps IMERG has to be shifted to map it to gencast on the same timestamp
            step_pred(int): Number of steps gen predictions time axis is to be shifted
            window_size(int): Number of timesteps to be considered in a slice
            scaling_type(str): Type of scaling - "none", "global_max", or "grid_wise_min_max"
            scaling_values: For global_max: single float value, for grid_wise_min_max: tuple of (min_values, max_values) arrays
            max_calc_source(str): Source for max calculation when using global_max - "pred" or "both"
        """
        super().__init__()
        self.file_mapping = mapping
        self.index = []
        self.scaling_type = scaling_type
        self.scaling_values = scaling_values
        self.max_calc_source = max_calc_source
        self.epsilon = 1e-8  # to prevent division by zero
        self.step_imerg = step_imerg
        self.step_pred = step_pred

        # Validate max_calc_source
        if self.scaling_type == "global_max" and self.max_calc_source not in ["pred", "both"]:
            raise ValueError("max_calc_source must be either 'pred' or 'both' when using global_max scaling")

        # Initialize scaling value tracking
        if self.scaling_values is None and self.scaling_type != "none":
            if self.scaling_type == "global_max":
                self.global_max = 0.0
            elif self.scaling_type == "grid_wise_min_max":
                self.min_values = None
                self.max_values = None

        # Process all files and calculate scaling values
        for imerg_file, pred_file in tqdm(self.file_mapping.items(), desc="Processing files",file=sys.stdout):
            self._process_file_pair(imerg_file, pred_file, window_size)

        # Set final scaling values
        if self.scaling_values is None and self.scaling_type != "none":
            if self.scaling_type == "global_max":
                self.scaling_values = self.global_max
                print(f"Calculated global max value: {self.global_max}")
            elif self.scaling_type == "grid_wise_min_max":
                self.scaling_values = (self.min_values, self.max_values)
                print("Calculated grid-wise min-max values")

        print(f"Training dataset created with {len(self.index)} pairs of adjacent timesteps for IMERG and GenCast")

    def _process_file_pair(self, imerg_file_path, pred_file_path, window_size):
        """Process and create index of imerg to corresponding prediction timestep."""
        try:
            with xr.open_dataset(imerg_file_path, decode_timedelta=True) as imerg_data, \
                 xr.open_dataset(pred_file_path, decode_timedelta=True) as pred_data:
                
                # Calculate scaling values if needed
                if self.scaling_values is None and self.scaling_type != "none":
                    self._update_scaling_values(imerg_data, pred_data)

                pred_times = pred_data.time.values
                if 'sample' in pred_data.coords:
                    for num_sample in range(pred_data.sample.size):
                        for i in range(len(pred_times)-window_size+1):
                            self.index.append({
                                'imerg_file': imerg_file_path,
                                'pred_file': pred_file_path,
                                'sample_index': num_sample,
                                'current_pred_idx': i + self.step_pred,
                                'next_pred_idx': i + window_size - 1 + self.step_pred,
                                'current_imerg_idx': self.step_imerg + i,
                                'next_imerg_idx': self.step_imerg + window_size + i - 1,
                                'date': imerg_data.time.values[self.step_imerg+i]
                            })
        except Exception as e:
            print(f"Error processing files {imerg_file_path} and {pred_file_path}: {e}")

    def _update_scaling_values(self, imerg_data, pred_data):
        """Update scaling values while processing files."""
        # Select the region of interest
        imerg = imerg_data.sel(lat=slice(5,40), lon=slice(60,100))
        pred = pred_data.sel(lat=slice(5,40), lon=slice(60,100))

        if self.scaling_type == "global_max":
            # Calculate max based on the specified source
            if self.max_calc_source == "pred":
                current_max = float(pred['total_precipitation_12hr'].max().values * 1000)  # Convert to mm
            else:  # "both"
                imerg_max = float(imerg.precipitation.max().values)
                pred_max = float(pred['total_precipitation_12hr'].max().values * 1000)  # Convert to mm
                current_max = max(imerg_max, pred_max)
            
            self.global_max = max(self.global_max, current_max)
        elif self.scaling_type == "grid_wise_min_max":
            # Convert to numpy arrays
            imerg_np = imerg.precipitation.values
            pred_np = pred['total_precipitation_12hr'].values * 1000  # Convert to mm

            # Calculate min and max for current file pair
            imerg_min = np.min(imerg_np, axis=0)
            imerg_max = np.max(imerg_np, axis=0)
            pred_min = np.min(pred_np, axis=(0,1,2))
            pred_max = np.max(pred_np, axis=(0,1,2))

            # Initialize arrays if first file
            if self.min_values is None:
                self.min_values = np.full_like(imerg_min, np.inf)
                self.max_values = np.full_like(imerg_max, -np.inf)

            # Update global min and max
            # self.min_values = np.minimum(self.min_values, np.minimum(imerg_min, pred_min))
            # self.max_values = np.maximum(self.max_values, np.maximum(imerg_max, pred_max))

            self.min_values = np.minimum(self.min_values, pred_min)
            self.max_values = np.maximum(self.max_values, pred_max)

    def __len__(self):
        """Return the size of the index i.e. the number of pieces of the datasets"""
        return len(self.index)

    def _normalize_data(self, data_tensor):
        """Apply normalization based on the specified scaling type."""
        if self.scaling_type == "none":
            return data_tensor
        elif self.scaling_type == "global_max":
            if self.scaling_values is not None:
                return data_tensor / (self.scaling_values + self.epsilon)
            return data_tensor
        elif self.scaling_type == "grid_wise_min_max":
            if self.scaling_values is not None:
                min_values, max_values = self.scaling_values
                # Ensure min_values and max_values have the right shape for broadcasting
                min_values = torch.tensor(min_values).unsqueeze(0).unsqueeze(0)  # Add time and channel dimensions
                max_values = torch.tensor(max_values).unsqueeze(0).unsqueeze(0)
                # Normalize to [0,1] range
                return (data_tensor - min_values) / (max_values - min_values + self.epsilon)
            return data_tensor
        else:
            raise ValueError(f"Unknown scaling type: {self.scaling_type}")

    def __getitem__(self, idx):
        """Returns the input predictions and corresponding IMERG data to be mapped to.

        Returns:
            torch.Tensor: Prediction data tensor
            torch.Tensor: IMERG data tensor
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_info = self.index[idx]
        
        with xr.open_dataset(sample_info['imerg_file'], decode_timedelta=True) as imerg_data, \
             xr.open_dataset(sample_info['pred_file'], decode_timedelta=True) as pred_data:
            
            if 'sample' in pred_data.coords:
                pred = pred_data.isel(
                    time=slice(sample_info['current_pred_idx'], sample_info['next_pred_idx']+1),
                    batch=0,
                    sample=sample_info['sample_index']
                ).sel(lat=slice(5,40), lon=slice(60,100)) * 1000
            else:
                pred = pred_data.isel(
                    time=slice(sample_info['current_pred_idx'], sample_info['next_pred_idx']+1),
                    batch=0
                ).sel(lat=slice(5,40), lon=slice(60,100)) * 1000

            imerg = imerg_data.isel(
                time=slice(sample_info['current_imerg_idx'], sample_info['next_imerg_idx']+1)
            ).transpose('time', 'lat', 'lon').sel(lat=slice(5,40), lon=slice(60,100))

            pred_tensor = torch.tensor(pred['total_precipitation_12hr'].values.astype(np.float32)).unsqueeze(1)
            imerg_tensor = torch.tensor(imerg.precipitation.values.astype(np.float32)).unsqueeze(1)

            # Apply normalization
            pred_tensor = self._normalize_data(pred_tensor)
            imerg_tensor = self._normalize_data(imerg_tensor)

            return pred_tensor, imerg_tensor
