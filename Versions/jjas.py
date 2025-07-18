import torch
from Models import RefinementModel
from Models import TrajGRU,UNet
from torchinfo import summary
from train_refinement_model import train_refinement_model
import time
import numpy as np
import random
from DatasetVariations import JJASDataset
from torch.utils.data import random_split,DataLoader,SubsetRandomSampler
from sklearn.model_selection import KFold
import mlflow

import threading
import os
import psutil
import io
from contextlib import redirect_stdout

class ResourceLogger:
    def __init__(self, logger, interval=10, log_file='/scratch/IITB/monsoon_lab/24d1236/pratham/Model/Logs/resources/k3b64noscaling.log'):
        self.logger = logger
        self.interval = interval
        self.log_file = log_file
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._log_loop, daemon=True)
        # Clear the log file at the start
        with open(self.log_file, 'w') as f:
            f.write('timestamp,cpu_percent,mem_mb,gpu_mem_allocated,gpu_mem_reserved,gpu_used_mem,gpu_util,pid,process_name\n')

    def start(self):
        self._stop_event.clear()
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self.thread.join()

    def _log_loop(self):
        process = psutil.Process(os.getpid())
        pid = os.getpid()
        while not self._stop_event.is_set():
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            cpu_percent = process.cpu_percent(interval=None)
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            gpu_mem_allocated = gpu_mem_reserved = gpu_used_mem = gpu_util = pname = ''
            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                try:
                    import subprocess
                    result = subprocess.run([
                        'nvidia-smi',
                        '--query-compute-apps=pid,process_name,used_memory,gpu_util',
                        '--format=csv,noheader,nounits'
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            cols = [c.strip() for c in line.split(',')]
                            if len(cols) >= 4 and str(pid) == cols[0]:
                                pname, gpu_used_mem, gpu_util = cols[1], cols[2], cols[3]
                                break
                except Exception as e:
                    pass
            log_line = f"{ts},{cpu_percent},{mem_mb},{gpu_mem_allocated},{gpu_mem_reserved},{gpu_used_mem},{gpu_util},{pid},{pname}\n"
            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(log_line)
            # Log to logger
            self.logger.info(f"[Resource] {log_line.strip()}")
            self._stop_event.wait(self.interval)

def jjas_main(logger,model,batch_size,scaling_type,hidden_channels,experiment_name,run_name,description,kernel_size=3):
    start_time = time.time()
    
    # Start resource logger
    # resource_logger = ResourceLogger(logger, interval=10,log_file=f'/scratch/IITB/monsoon_lab/24d1236/pratham/Model/Logs/resources/k{kernel_size}b{batch_size}maxscaling.log')
    # resource_logger.start()
    try:
        logger.info(f"Run desc: {description}")
        logger.info("Starting data preparation...")
        PRED_DIR='/scratch/IITB/monsoon_lab/24d1236/pratham/gencast_1deg/predictions_2018/'
        IMERG_PATH='/scratch/IITB/monsoon_lab/24d1236/pratham/Datasets/june_sept_2018/IMERG/IMERG1_from_31May2018_to_06Oct2018_resampled_12hr_final.nc'
        
        dataset = JJASDataset(PRED_DIR,IMERG_PATH,2,scaling_type=scaling_type)

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
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, 
                                worker_init_fn=seed_worker, generator=loader_gen)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=True,
                                worker_init_fn=seed_worker, generator=loader_gen)
        logger.info(f"Training indices:{train_dataset.indices}")
        logger.info("Data loaders created. Initializing model...")
        
        # Print model summary
        logger.info("Model summary:")
        # Capture summary output
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            summary(model, input_size=(1,2, 1, 36, 41))

        # Log the summary
        logger.info("Model Summary:\n%s", buffer.getvalue()) 
        logger.info("Starting model training...")
        
        # Train model with MLflow tracking
        model, optimizer, history, num_epochs = train_refinement_model(
            model, 
            logger,
            train_loader, 
            val_loader, 
            num_epochs=50,
            # experiment_name=f"JJAS_{scaling_type}_h{'_'.join(map(str,hidden_channels))}",
            experiment_name=experiment_name,
            # experiment_name=f"Traj_JJAS_{scaling_type}",
            # experiment_name=f"Unet_JJAS_{scaling_type}",
            # run_name=f"1_run_b{batch_size}_k{kernel_size}_h{'_'.join(map(str,hidden_channels))}",
            # run_name=f"1_Traj_run_b{batch_size}",
            run_name=run_name,
            # description=f"First run with kernel size {kernel_size} and batch size {batch_size} with hidden channels as {'_'.join(map(str,hidden_channels))}",
            # description=f"Traj First Run batch size as {batch_size}",
            description=description,
            log_file=f'/scratch/IITB/monsoon_lab/24d1236/pratham/Model/model_training.log'
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Training completed. Total execution time: {elapsed_time:.4f} seconds")
    finally:
        print('completed')
        # resource_logger.stop()
        # Log resource stats file as MLflow artifact
        # mlflow.log_artifact(resource_logger.log_file)

def jjas_kfold(logger, k_folds=5, kernel_size=3, num_epochs=50):
    """
    Perform k-fold cross validation on the JJAS dataset.
    
    Args:
        logger: Logger instance for logging
        k_folds (int): Number of folds for cross validation (default: 5)
        kernel_size (int): Kernel size for the model (default: 3)
        num_epochs (int): Number of epochs per fold (default: 50)
    """
    start_time = time.time()
    
    # Start resource logger
    resource_logger = ResourceLogger(logger, interval=10)
    resource_logger.start()
    try:
        logger.info("Starting data preparation...")
        PRED_DIR='/scratch/IITB/monsoon_lab/24d1236/pratham/gencast_1deg/predictions'
        IMERG_PATH='/scratch/IITB/monsoon_lab/24d1236/pratham/Model/june_sept_2014/IMERG/IMERG1_from_31May2018_to_06Oct2018_resampled_12hr_final.nc'
        
        dataset = JJASDataset(PRED_DIR,IMERG_PATH,2,scaling_type='NoScaling')

        logger.info("Data preparation completed. Setting up k-fold cross validation...")
        
        # Set worker seed for reproducibility
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Initialize k-fold cross validation
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Store results for each fold
        fold_results = []
        
        # Get dataset indices for k-fold splitting
        dataset_indices = list(range(len(dataset)))
        
        logger.info(f"Starting {k_folds}-fold cross validation...")
        
        for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset_indices)):
            logger.info(f"Training Fold {fold + 1}/{k_folds}")
            logger.info(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
            
            # Create samplers for this fold
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            
            # Create data loaders for this fold
            train_loader = DataLoader(
                dataset, 
                batch_size=64, 
                sampler=train_sampler,
                worker_init_fn=seed_worker
            )
            val_loader = DataLoader(
                dataset, 
                batch_size=64, 
                sampler=val_sampler,
                worker_init_fn=seed_worker
            )
            
            # Initialize model for this fold
            model = RefinementModel(kernel_size=kernel_size)
            
            # Print model summary for first fold only
            if fold == 0:
                logger.info("Model summary:")
                summary(model, (1, 2, 1, 36, 41))
            
            logger.info(f"Starting training for fold {fold + 1}...")
            
            # Train model for this fold
            fold_model, optimizer, history, epochs_trained = train_refinement_model(
                model, 
                logger,
                train_loader, 
                val_loader, 
                num_epochs=num_epochs,
                experiment_name=f"JJAS_No_Scaling_KFold",
                run_name=f"fold_{fold + 1}_kernel_size_{kernel_size}",
                description=f"Fold {fold + 1} of {k_folds}-fold CV on JJAS no scaling with kernel size {kernel_size}",
            )
            
            # Store fold results
            fold_result = {
                'fold': fold + 1,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'best_val_loss': min(history['val_loss']) if history['val_loss'] else float('inf'),
                'best_train_loss': min(history['train_loss']) if history['train_loss'] else float('inf'),
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else float('inf'),
                'epochs_trained': epochs_trained,
                'history': history
            }
            fold_results.append(fold_result)
            
            logger.info(f"Fold {fold + 1} completed. Best val loss: {fold_result['best_val_loss']:.6f}")
        
        # Calculate and log cross-validation summary
        val_losses = [result['best_val_loss'] for result in fold_results]
        train_losses = [result['best_train_loss'] for result in fold_results]
        
        mean_val_loss = np.mean(val_losses)
        std_val_loss = np.std(val_losses)
        mean_train_loss = np.mean(train_losses)
        std_train_loss = np.std(train_losses)
        
        logger.info("=" * 50)
        logger.info("K-FOLD CROSS VALIDATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Number of folds: {k_folds}")
        logger.info(f"Kernel size: {kernel_size}")
        logger.info(f"Epochs per fold: {num_epochs}")
        logger.info(f"Mean validation loss: {mean_val_loss:.6f} ± {std_val_loss:.6f}")
        logger.info(f"Mean training loss: {mean_train_loss:.6f} ± {std_train_loss:.6f}")
        logger.info(f"Best validation loss: {min(val_losses):.6f}")
        logger.info(f"Worst validation loss: {max(val_losses):.6f}")
        
        # Log individual fold results
        logger.info("\nIndividual fold results:")
        for result in fold_results:
            logger.info(f"Fold {result['fold']}: Val Loss = {result['best_val_loss']:.6f}, Train Loss = {result['best_train_loss']:.6f}")
        
        # Log k-fold conclusion to MLflow
        mlflow.set_experiment("JJAS_No_Scaling_KFold")
        with mlflow.start_run(run_name="kfold conclusion"):
            mlflow.set_tag("mlflow.note.content", "K-fold cross validation summary results")
            mlflow.log_param("k_folds", k_folds)
            mlflow.log_param("kernel_size", kernel_size)
            mlflow.log_param("num_epochs_per_fold", num_epochs)
            mlflow.log_metric("mean_val_loss", mean_val_loss)
            mlflow.log_metric("std_val_loss", std_val_loss)
            mlflow.log_metric("mean_train_loss", mean_train_loss)
            mlflow.log_metric("std_train_loss", std_train_loss)
            mlflow.log_metric("best_val_loss", min(val_losses))
            mlflow.log_metric("worst_val_loss", max(val_losses))
            
            # Log individual fold results
            for i, result in enumerate(fold_results):
                mlflow.log_metric(f"fold_{i+1}_best_val_loss", result['best_val_loss'])
                mlflow.log_metric(f"fold_{i+1}_best_train_loss", result['best_train_loss'])
            
            logger.info("K-fold cross validation results logged to MLflow run 'kfold conclusion'")
            # Log resource stats file as MLflow artifact
            mlflow.log_artifact(resource_logger.log_file)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"K-fold cross validation completed. Total execution time: {elapsed_time:.4f} seconds")
        
        return fold_results, {
            'mean_val_loss': mean_val_loss,
            'std_val_loss': std_val_loss,
            'mean_train_loss': mean_train_loss,
            'std_train_loss': std_train_loss,
            'best_val_loss': min(val_losses),
            'worst_val_loss': max(val_losses)
        }
    finally:
        resource_logger.stop()

