import os
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from glob import glob
import xarray as xr
import pandas as pd
from DatasetVariations import TrainDataset,T4DatasetNoScaling,T4DatasetMinMaxScalerOverall,T4DatasetNoScaling
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from Models import RefinementModel
from torchinfo import summary
from train_refinement_model import train_refinement_model
from file_mappings import T_6_file_to_file_mapping_training, T_4_file_to_file_mapping_training, T_4_file_to_file_mapping_validation, T_8_file_to_file_mapping_training
import time
import mlflow
import mlflow.pytorch
import numpy as np
import random
from logger_config import logger

# Set MLflow tracking URI (optional - if you want to use a specific server)
# mlflow.set_tracking_uri("http://localhost:5000")  # Uncomment if using MLflow server

def main(logger):
    start_time = time.time()
    
    logger.info("Starting data preparation...")
    
    # Data preparation
    # t4training_data = T4DatasetNoScaling(
    #     T_4_file_to_file_mapping_training, 
    #     data='forecast', 
    # )
    # t6training_data = T4DatasetNoScaling(
    #     T_6_file_to_file_mapping_training, 
    #     data='forecast', 
    #     step=
    # )
    t4training_data_no_scaling = TrainDataset(
        T_4_file_to_file_mapping_training, 
        step_imerg=25,
        step_pred=0,
        window_size=2,
        scaling_type="none",
        scaling_values=None,
        max_calc_source="both"
    )
    t6training_data_no_scaling = TrainDataset(
        T_6_file_to_file_mapping_training, 
        step_imerg=21,
        step_pred=0,
        window_size=2,
        scaling_type="none",
        scaling_values=None,
        max_calc_source="both"
    )
    t8val_data_no_scaling = TrainDataset(
        T_8_file_to_file_mapping_training, 
        step_imerg=17,
        step_pred=0,
        window_size=2,
        scaling_type="none",
        scaling_values=None,
        max_calc_source="both"
    )
    training_data = ConcatDataset([t4training_data_no_scaling,t6training_data_no_scaling])
    
    logger.info("Data preparation completed. Creating data loaders...")
    
    # Set worker seed for reproducibility
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)  # Set a fixed seed for the generator
    
    train_loader = DataLoader(training_data, 64, shuffle=True, 
                            worker_init_fn=seed_worker,
                            generator=g)
    val_loader = DataLoader(t8val_data_no_scaling, 64, shuffle=True,
                            worker_init_fn=seed_worker,
                            generator=g)
    
    logger.info("Data loaders created. Initializing model...")
    
    model = RefinementModel()
    
    # Print model summary
    logger.info("Model summary:")
    summary(model, (1, 2, 1, 36, 41))
    
    logger.info("Starting model training...")
    
    # Train model with MLflow tracking
    model, optimizer, history, num_epochs = train_refinement_model(
        model, 
        logger,
        train_loader, 
        val_loader, 
        num_epochs=50,
        experiment_name="Without_Scaling",
        run_name="1_first_run"
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Training completed. Total execution time: {elapsed_time:.4f} seconds")
    
if __name__ == "__main__":
    main(logger)
