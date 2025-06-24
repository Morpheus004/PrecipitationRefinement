import torch
from Models import RefinementModel
from torchinfo import summary
from train_refinement_model import train_refinement_model
import time
import numpy as np
import random
from DatasetVariations import JJASDataset
from torch.utils.data import random_split,DataLoader

def jjas_main(logger,kernel_size=3):
    start_time = time.time()
    
    logger.info("Starting data preparation...")
    PRED_DIR='/scratch/IITB/monsoon_lab/24d1236/pratham/gencast_1deg/predictions'
    IMERG_PATH='/scratch/IITB/monsoon_lab/24d1236/pratham/Model/june_sept_2014/IMERG/IMERG1_from_31May2018_to_06Oct2018_resampled_12hr_final.nc'
    
    dataset = JJASDataset(PRED_DIR,IMERG_PATH,2,scaling_type='NoScaling')

    logger.info("Data preparation completed. Creating data loaders...")
    
    # Set worker seed for reproducibility
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Generator for dataset splitting
    split_gen = torch.Generator()
    split_gen.manual_seed(42)

    # Generator for DataLoader operations
    loader_gen = torch.Generator()
    loader_gen.manual_seed(42)  # Same seed is fine, or use different seeds

    # Split with dedicated generator
    train_dataset, val_dataset = random_split(
        dataset, [0.8, 0.2], generator=split_gen
    )

    # DataLoaders with dedicated generator
    train_loader = DataLoader(train_dataset, 64, shuffle=True, 
                            worker_init_fn=seed_worker, generator=loader_gen)
    val_loader = DataLoader(val_dataset, 64, shuffle=True,
                            worker_init_fn=seed_worker, generator=loader_gen)
    logger.info(f"Training indices:{train_dataset.indices}")
    logger.info("Data loaders created. Initializing model...")
    
    model = RefinementModel(kernel_size=kernel_size)
    
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
        experiment_name="JJAS_No_Scaling",
        run_name="2_run_kernel_size_5",
        description="2 run on JJAS no scaling and kernel size is 5"
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Training completed. Total execution time: {elapsed_time:.4f} seconds")
    

