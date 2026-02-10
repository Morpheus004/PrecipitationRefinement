import os
import typer
import subprocess
from typing import Optional
import xarray as xr
import numpy as np
from pathlib import Path
import tempfile
import uuid
import shutil

app = typer.Typer()

def select_file_with_fzf(prompt_text: str = "Select a file:") -> str:
    """
    Use fzf to interactively select a file.
    Returns the selected file path or empty string if canceled.
    """
    try:
        # Check if fzf and fd are available
        if shutil.which("fzf") is None or shutil.which("fd") is None:
            typer.echo("fzf or fd not found. Please install them to use interactive file selection.")
            return ""
        
        # Run fd to list files and pipe to fzf for selection
        # Using fd with -t f to only list files (not directories)
        cmd = "fd -t f /newstor/vishald/ug1_monsoonlab/pratham . | fzf --height 40% --reverse --prompt='{}' --preview 'cat {}'".format(prompt_text, "{}")
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return ""
    except Exception as e:
        typer.echo(f"Error selecting file: {e}")
        return ""

@app.command()
def convert(
    input_file: str = typer.Option(
        "", 
        "--input-file", "-i", 
        help="Path to input NetCDF file with 0.25° resolution",
        prompt="Path to input NetCDF file"
    ),
    output_file: Optional[str] = typer.Option(None, "--output-file", "-o", help="Path to output file. If not provided, will be placed in same directory with '1deg_' prefix"),
    temp_dir: Optional[str] = typer.Option(None, "--temp-dir", "-t", help="Directory for temporary files"),
    variables: Optional[str] = typer.Option(None, "--variables", "-v", help="Comma-separated list of variables to process (default: all)"),
    reference_file: Optional[str] = typer.Option('/scratch/vishald/ug1_monsoonlab/pratham/cli_tool/source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc', "--reference-file", "-r", help="Reference file for land_sea_mask and geopotential_at_surface"),
    grid_file: str = typer.Option("grid.txt", "--grid-file", "-g", help="Path to the grid file for CDO remapping"),
    use_fzf: bool = typer.Option(True, "--use-fzf/--no-fzf", help="Use fzf for interactive file selection")
):
    """
    Convert a NetCDF file from 0.25° resolution to 1° resolution.
    
    This tool transforms NetCDF files from 0.25° to 1° spatial resolution 
    using CDO remapping, preserving coordinate information and adding 
    required variables.
    """
    # Use fzf for file selection if requested and input_file is not provided
    if use_fzf and not input_file:
        selected_file = select_file_with_fzf("Select input NetCDF file:")
        if selected_file:
            input_file = selected_file
        else:
            typer.echo("No file selected. Exiting.")
            raise typer.Exit(1)
    
    # If still no input file, exit
    if not input_file:
        typer.echo("Error: Input file is required")
        raise typer.Exit(1)
    
    typer.echo(f"Processing {input_file}")
    
    # Create temp directory if not provided
    use_temp_dir = temp_dir is None
    if use_temp_dir:
        # Use the same directory as input file and create a temp subdir
        input_dir = os.path.dirname(os.path.abspath(input_file))
        temp_dir_name = f"temp_{uuid.uuid4().hex[:8]}"
        temp_dir = os.path.join(input_dir, temp_dir_name)
        os.makedirs(temp_dir, exist_ok=True)
        typer.echo(f"Using temporary directory: {temp_dir}")
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    # Generate default output path if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"1deg_{input_path.name}")
    
    # Check if grid file exists
    if not os.path.exists(grid_file):
        typer.echo(f"Error: Grid file {grid_file} not found")
        raise typer.Abort()
    
    try:
        # Open dataset
        df = xr.open_dataset(input_file)
        
        # Create intermediate file with selected variables
        intermediate_file = os.path.join(temp_dir, "selected_vars.nc")
        
        if variables:
            var_list = [var.strip() for var in variables.split(",")]
            typer.echo(f"Selected variables: {var_list}")
            selected_vars = df[var_list]
        else:
            # Default variables from your example
            default_vars = [
                '2m_temperature', 'sea_surface_temperature', 'mean_sea_level_pressure',
                '10m_v_component_of_wind', 'total_precipitation_12hr', '10m_u_component_of_wind',
                'u_component_of_wind', 'specific_humidity', 'temperature', 
                'vertical_velocity', 'v_component_of_wind', 'geopotential'
            ]
            
            # Filter to only include variables that exist in the dataset
            available_vars = [var for var in default_vars if var in df]
            typer.echo(f"Processing variables: {available_vars}")
            
            if not available_vars:
                typer.echo("No matching variables found in dataset, using all variables")
                selected_vars = df
            else:
                selected_vars = df[available_vars]
        
        # Handle batch dimension if it exists
        if 'batch' in selected_vars.dims:
            selected_vars = selected_vars.isel(batch=0)
        
        typer.echo(f"Saving intermediate file: {intermediate_file}")
        selected_vars.to_netcdf(intermediate_file)
        
        # Run CDO to remap to 1 degree resolution
        temp_output = os.path.join(temp_dir, "remapped.nc")
        typer.echo("Running CDO remapping...")
        cdo_cmd = f"cdo remapcon,{grid_file} {intermediate_file} {temp_output}"
        typer.echo(f"Executing: {cdo_cmd}")
        
        result = subprocess.run(cdo_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            typer.echo("CDO Error:")
            typer.echo(result.stderr)
            raise typer.Abort()
        
        # Load remapped data
        final_data = xr.open_dataset(temp_output)
        
        # Add land_sea_mask and geopotential_at_surface if reference file is provided
        if reference_file:
            typer.echo(f"Adding reference data from {reference_file}")
            ref_data = xr.open_dataset(reference_file)
            
            if 'land_sea_mask' in ref_data:
                final_data['land_sea_mask'] = ref_data.land_sea_mask
            
            if 'geopotential_at_surface' in ref_data:
                final_data['geopotential_at_surface'] = ref_data.geopotential_at_surface
        
        # Handle dimensions and coordinates
        if 'batch' in df.dims:
            final_data = final_data.expand_dims(dim='batch')
        
        # Handle datetime coordinates if they exist
        if 'datetime' in df.coords and 'time' in df.dims:
            time_size = df.sizes.get('time', 0)
            if time_size > 0:
                # Calculate time values
                time_values = np.arange(0, time_size * 12 * 3600 * 10**9, 12 * 3600 * 10**9, dtype="timedelta64[ns]")
                
                # Assign coordinates
                final_data = final_data.assign_coords(time=('datetime', time_values))
                final_data = final_data.swap_dims({'datetime': 'time'})
                
                # Get original datetime values
                valid_time_values = df["datetime"].values
                batch_size = 1
                if 'batch' in df.dims:
                    batch_size = df.sizes['batch']
                
                # Reshape to match expected dimensions
                valid_time_array = xr.DataArray(valid_time_values.reshape(batch_size, time_size), 
                                              dims=("batch", "time"))
                final_data = final_data.assign_coords({'datetime': valid_time_array})
        
        # Add derived variables if possible
        try:
            from graphcast import data_utils
            typer.echo("Adding derived variables...")
            data_utils.add_derived_vars(final_data)
            # Remove day and year progress variables
            if 'day_progress' in final_data:
                final_data = final_data.drop_vars(['day_progress'])
            if 'year_progress' in final_data:
                final_data = final_data.drop_vars(['year_progress'])
        except ImportError:
            typer.echo("graphcast module not found, skipping derived variables")
        
        # Save final output
        typer.echo(f"Saving output to {output_file}")
        final_data.to_netcdf(output_file)
        typer.echo(f"✅ Conversion completed successfully")
        
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise
    
    finally:
        # Clean up temporary directory if we created it
        if use_temp_dir and os.path.exists(temp_dir):
            import shutil
            typer.echo(f"Cleaning up temporary files")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    app()
