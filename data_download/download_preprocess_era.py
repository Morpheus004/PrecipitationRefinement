import typer
import os
from datetime import datetime, timedelta
import calendar
import cdsapi
import concurrent.futures
from typing import Dict, List, Tuple
from tqdm import tqdm
import zipfile  # Added for unzipping files
import xarray as xr  # Added for NetCDF processing
import numpy as np  # Added for array operations
import pandas as pd
import glob  # Added for file searching

app = typer.Typer()
def get_days_in_range(start_date: str, end_date: str) -> Dict[Tuple[int, int], List[str]]:
    """
    Generate a dictionary of months with their corresponding days within the date range.
    
    Args:
    start_date (str): Start date in YYYY-MM-DD format
    end_date (str): End date in YYYY-MM-DD format
    
    Returns:
    dict: A dictionary with (year, month) tuples as keys and lists of days as values
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    months_days = {}
    
    # Iterate through each month in the range
    current = start.replace(day=1)  # Always start at the first day of the month
    while current <= end:
        month_key = (current.year, current.month)
        
        # Determine the first and last day for this month
        if current.year == start.year and current.month == start.month:
            first_day = start.day
        else:
            first_day = 1
        
        if current.year == end.year and current.month == end.month:
            last_day = end.day
        else:
            last_day = calendar.monthrange(current.year, current.month)[1]
        
        # Generate list of days as zero-padded strings
        month_days = [str(day).zfill(2) for day in range(first_day, last_day + 1)]
        months_days[month_key] = month_days
        
        # Move to the next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return months_days

def unzip_and_rename_files(zip_file_path: str) -> None:
    """
    Unzip the single level files and rename them based on the original zip filename.
    
    Args:
    zip_file_path (str): Path to the zip file
    """
    try:
        # Get the base name of the zip file (without extension)
        base_name = os.path.splitext(os.path.basename(zip_file_path))[0]
        # Get the directory of the zip file
        extract_dir = os.path.dirname(zip_file_path)
        
        print(f"Unzipping {zip_file_path}...")
        
        # Extract all files from the zip
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
            # For each file in the zip
            for file_info in zip_ref.infolist():
                # Get the extracted file path
                extracted_file = os.path.join(extract_dir, file_info.filename)
                
                # Create a new name based on original zip file and the extracted file's name
                if "instant" in file_info.filename:
                    new_name = f"{base_name}_instant.nc"
                elif "accum" in file_info.filename:
                    new_name = f"{base_name}_accum.nc"
                else:
                    # Keep original name if not matching known patterns
                    continue
                
                new_path = os.path.join(extract_dir, new_name)
                
                # Rename the file
                if os.path.exists(extracted_file):
                    os.rename(extracted_file, new_path)
                    print(f"Renamed {file_info.filename} to {new_name}")
                
        print(f"Successfully unzipped and renamed files from {zip_file_path}")
        
    except Exception as e:
        print(f"Error unzipping or renaming files from {zip_file_path}: {e}")

def process_era5_for_gencast(output_dir: str, start_date: str, end_date: str) -> str:
    """
    Process ERA5 data files for the entire date range to create a single NetCDF file
    suitable for input to the GenCast model.
    
    Args:
    output_dir (str): Directory containing the ERA5 data files
    start_date (str): Start date in YYYY-MM-DD format
    end_date (str): End date in YYYY-MM-DD format
    
    Returns:
    str: Path to the output NetCDF file
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        date_range_str = f"{start_dt.strftime('%Y%m%d')}-{end_dt.strftime('%Y%m%d')}"
        output_file = os.path.join(output_dir, f"gencast_input_{date_range_str}.nc")
        
        print(f"Processing ERA5 data from {start_date} to {end_date} for GenCast input...")
        
        # Find all relevant files in the output directory
        sl_accum_files = sorted(glob.glob(os.path.join(output_dir, "era5_single_levels_*_accum.nc")))
        sl_instant_files = sorted(glob.glob(os.path.join(output_dir, "era5_single_levels_*_instant.nc")))
        pl_files = sorted(glob.glob(os.path.join(output_dir, "era5_pressure_levels_*.nc")))
        
        if not sl_accum_files or not sl_instant_files or not pl_files:
            print("Error: Could not find all required ERA5 files in the output directory.")
            return None
        
        print(f"Found {len(sl_accum_files)} accumulated files, {len(sl_instant_files)} instantaneous files, and {len(pl_files)} pressure level files.")
        
        # Open and concatenate all accumulated files
        print("Concatenating accumulated files...")
        sl_accum_datasets = [xr.open_dataset(file) for file in sl_accum_files]
        slp = xr.concat(sl_accum_datasets, dim='valid_time')
        
        # Resample accumulated precipitation to 12-hour intervals
        slp_12hrp = slp.resample(valid_time='12h', label='right', closed='right').sum()
        
        # Open and concatenate all instantaneous files
        print("Concatenating instantaneous files...")
        sl_instant_datasets = [xr.open_dataset(file) for file in sl_instant_files]
        slo = xr.concat(sl_instant_datasets, dim='valid_time')
        
        # Open and concatenate all pressure level files
        print("Concatenating pressure level files...")
        pl_datasets = [xr.open_dataset(file) for file in pl_files]
        pl = xr.concat(pl_datasets, dim='valid_time')
        
        # Rename geopotential in pressure level data
        pl = pl.rename({'z': 'Geopotential'})
        
        # Merge datasets
        print("Merging datasets...")
        data = xr.merge([slp_12hrp.isel(valid_time=slice(2, None)), slo, pl], join='inner')
        
        # Remove unnecessary coordinates
        data = data.reset_coords(['number', 'expver'], drop=True)
        
        # Add batch dimension
        data_dim = data.expand_dims(dim='batch')
        
        # Rename variables to match GenCast expectations
        data_dim = data_dim.rename({
            'Geopotential': 'geopotential',
            'valid_time': 'datetime',
            'longitude': 'lon',
            'latitude': 'lat',
            'pressure_level': 'level',
            'u10': '10m_u_component_of_wind',
            'v10': '10m_v_component_of_wind',
            't2m': '2m_temperature',
            'msl': 'mean_sea_level_pressure',
            'tp': 'total_precipitation_12hr',
            'sst': 'sea_surface_temperature',
            'q': 'specific_humidity',
            't': 'temperature',
            'u': 'u_component_of_wind',
            'v': 'v_component_of_wind',
            'w': 'vertical_velocity'
            # 'z': 'geopotential'
        })
        
        # Create time values based on the actual length of datetime dimension
        # This code was a problem because this make time values in ns so high that it exceeded integer contraints
#        time_values_for_n_days = np.arange(
#            0, 
#            len(data_dim.datetime) * 12 * 3600 * 10**9, 
#            12 * 3600 * 10**9, 
#            dtype="timedelta64[ns]"
#        )
        
        time_values_for_n_days = pd.date_range(
            start=start_dt + timedelta(hours=24),  # since you used `slice(2, None)` for 12hr summed data
            periods=len(data_dim.datetime),
            freq='12H'
        ).to_numpy(dtype='datetime64[ns]')
        
        # Get sizes for reshaping
        batch_size = data_dim.sizes['batch']
        time_size = data_dim.sizes['datetime']
        
        print(f"Processing {time_size} time steps for GenCast input...")
        
        # Assign time coordinates
        rough = data_dim.assign_coords(time=('datetime', time_values_for_n_days))
        
        # Swap dimensions
        rough = rough.swap_dims({'datetime': 'time'})
        
        # Extract datetime values and reshape
        valid_time_values = rough["datetime"].values
        valid_time_array = xr.DataArray(
            valid_time_values.reshape(1, time_size), 
            dims=("batch", "time")
        )
        
        # Assign coordinates and save
        rough = rough.assign_coords({'datetime': valid_time_array})
        # rough.drop_vars(['lsm','z'])
        rough.to_netcdf(output_file)
        
        # Clean up open datasets
        for ds in sl_accum_datasets + sl_instant_datasets + pl_datasets:
            ds.close()
        
        print(f"Successfully created GenCast input file: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error processing ERA5 data for GenCast: {e}")
        return None

def retrieve_data(
    data_type: str, 
    year: int, 
    month: int, 
    days: List[str], 
    output_dir: str
) -> None:
    """
    Retrieve data for a specific month and data type.
    
    Args:
    data_type (str): Type of data ('pressure_levels' or 'single_levels')
    year (int): Year of data retrieval
    month (int): Month of data retrieval
    days (List[str]): List of days to retrieve
    output_dir (str): Directory to save output files
    """
    try:
        # Initialize client 
        client = cdsapi.Client()

        print(f"Processing {data_type} for {year}-{month:02d}, days: {days}")

        if data_type == 'pressure_levels':
            request = {
                "product_type": "reanalysis",
                "variable": [
                    "geopotential",
                    "specific_humidity",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "vertical_velocity"
                ],
                "year": str(year),
                "month": str(month).zfill(2),
                "day": days,
                "time": ["00:00", "12:00"],
                "pressure_level": [
                    "50", "100", "150", "200", "250", "300",
                    "400", "500", "600", "700", "850", "925", "1000"
                ],
                "data_format":"netcdf",
                "download_format":"unarchived"
            }
            dataset = "reanalysis-era5-pressure-levels"
            output_file = os.path.join(output_dir, f"era5_pressure_levels_{year}{month:02d}.nc")
        
        else:  # single levels
            request = {
                "product_type": "reanalysis",
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_temperature",
                    "mean_sea_level_pressure",
                    "sea_surface_temperature",
                    "total_precipitation",
                    "geopotential",
                    "land_sea_mask"
                ],
                "year": str(year),
                "month": str(month).zfill(2),
                "day": days,
                "time": [f"{hour:02d}:00" for hour in range(24)],
                "data_format":"netcdf",
                "download_format":"unarchived"
            }
            dataset = "reanalysis-era5-single-levels"
            output_file = os.path.join(output_dir, f"era5_single_levels_{year}{month:02d}.zip")
        
        print(f"Retrieving {data_type} data for {year}-{month:02d}")
        
        # Retrieve and download the data
        client.retrieve(dataset, request, output_file)
        print(f"Saved {data_type} data to {output_file}")
        
        # Unzip and rename single level files after download
        if data_type == 'single_levels':
            unzip_and_rename_files(output_file)
    
    except Exception as e:
        print(f"Error retrieving {data_type} data for {year}-{month:02d}: {e}")

@app.command()
def fetch(
    start_date: str = typer.Option(..., prompt="Enter start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., prompt="Enter end date (YYYY-MM-DD)"),
    download_dir: str = typer.Option(..., prompt="Enter name for output directory"),
    max_workers: int = typer.Option(default=4, help="Maximum number of concurrent workers"),
    process_for_gencast: bool = typer.Option(default=False, help="Process data for GenCast model after download")
):
    """
    Fetch ERA5 reanalysis data from Copernicus Climate Data Store (CDS) 
    for a multi-month date range using parallel processing.
    """
    # Validate date format and range
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start > end:
            print("Start date must be before or equal to end date.")
            raise typer.Abort()
    except ValueError:
        print("Invalid date format. Use YYYY-MM-DD.")
        raise typer.Abort()

    # Get days for each month in the range
    months_days = get_days_in_range(start_date, end_date)
    print(f"Months and days to process: {months_days}")

    # Ensure output directory exists
    output_dir = f"{download_dir}"
    os.makedirs(output_dir, exist_ok=True)

    # Use tqdm to show overall progress
    with tqdm(total=len(months_days)*2, desc="Data Retrieval Progress") as overall_progress:
        # Use ProcessPoolExecutor for more reliable parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Prepare and submit tasks for each month and data type
            futures = []
            for (year, month), days in months_days.items():
                # Submit pressure levels task
                pressure_future = executor.submit(
                    retrieve_data, 
                    'pressure_levels', 
                    year, 
                    month, 
                    days, 
                    output_dir
                )
                futures.append((pressure_future, overall_progress))

                # Submit single levels task
                single_future = executor.submit(
                    retrieve_data, 
                    'single_levels', 
                    year, 
                    month, 
                    days, 
                    output_dir
                )
                futures.append((single_future, overall_progress))

            # Wait for all tasks to complete and check for errors
            for future, progress_bar in futures:
                try:
                    future.result()  # This will raise any exceptions that occurred
                    progress_bar.update(1)
                except Exception as e:
                    print(f"Task failed: {e}")
                    progress_bar.update(1)

    print("Multi-month data retrieval completed.")
    
    # Process data for GenCast if requested
    if process_for_gencast:
        print("Processing downloaded data for GenCast model...")
        
        # Process all data into a single GenCast input file
        output_file = process_era5_for_gencast(output_dir, start_date, end_date)
        
        if output_file:
            print(f"Successfully created GenCast input file: {output_file}")

if __name__ == "__main__":
    app()
