from Versions import jjas_no_scaling
from logger_config import logger

if __name__=='__main__':
    # --- Process-dependent CPU and GPU Logging ---
    # import os
    # import psutil
    # import torch
    # import logging
    # 
    # process = psutil.Process(os.getpid())
    # cpu_percent = process.cpu_percent(interval=1)
    # mem_info = process.memory_info()
    # mem_mb = mem_info.rss / 1024 / 1024
    # logger.info(f"[Resource] CPU usage: {cpu_percent}%")
    # logger.info(f"[Resource] Memory usage: {mem_mb:.2f} MB")
    # 
    # if torch.cuda.is_available():
    #     gpu_mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
    #     gpu_mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024
    #     logger.info(f"[Resource] GPU memory allocated: {gpu_mem_allocated:.2f} MB")
    #     logger.info(f"[Resource] GPU memory reserved: {gpu_mem_reserved:.2f} MB")
    #     try:
    #         import subprocess
    #         result = subprocess.run([
    #             'nvidia-smi',
    #             '--query-gpu=utilization.gpu',
    #             '--format=csv,noheader,nounits'
    #         ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #         if result.returncode == 0:
    #             gpu_util = result.stdout.strip()
    #             logger.info(f"[Resource] GPU utilization: {gpu_util}%")
    #     except Exception as e:
    #         logger.warning(f"[Resource] Could not query GPU utilization: {e}")
    # else:
    #     logger.info("[Resource] CUDA not available. Skipping GPU logging.")

    jjas_no_scaling.jjas_main(logger=logger,kernel_size=5)
    
    # K-Fold Cross Validation Usage
    # Uncomment the following line to run k-fold cross validation
    # 
    # # Run 5-fold cross validation with kernel size 5
    # fold_results, summary_stats = jjas_no_scaling.jjas_kfold(
    #     logger=logger, 
    #     k_folds=5, 
    #     kernel_size=5, 
    #     num_epochs=50
    # )
    # 
    # # Results are automatically logged to MLflow run "kfold conclusion"
    # print(f"Mean validation loss: {summary_stats['mean_val_loss']:.6f}")
    # print(f"Best validation loss: {summary_stats['best_val_loss']:.6f}")
