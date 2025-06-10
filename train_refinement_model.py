import torch
import torch.nn as nn
from tqdm import tqdm
import os
import mlflow
import mlflow.pytorch
import numpy as np
import random
import copy
from logger_config import logger
import tempfile

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_refinement_model(model,logger, train_loader, val_loader=None, 
                          num_epochs=30, learning_rate=0.001,
                          device='cuda' if torch.cuda.is_available() else 'cpu', 
                          already_trained=False, checkpoint_path=None,
                          experiment_name="precipitation_refinement", run_name=None,
                          seed=42):
    """
    Training function for the precipitation refinement model with MLflow tracking
    
    Args:
        model: PrecipitationRefinementModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run training on ('cuda' or 'cpu')
        already_trained: Bool to say if model has already been trained
        checkpoint_path: Path to the pth file which will have history, optimizer state, number of epochs trained and model state_dict.
        experiment_name: Name of the MLflow experiment
        run_name: Name of the MLflow run (optional)
        seed: Random seed for reproducibility (default: 42)
    Returns:
        model,optimizer,history,num_epochs
    """
    
    # Set random seed for reproducibility
    set_seed(seed)
    
    torch.use_deterministic_algorithms(True)

    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    mlflow.enable_system_metrics_logging()
    
    with mlflow.start_run(run_name=run_name,log_system_metrics=True):
        logger.info(f'Using device: {device}')
        model = model.to(device)
        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        start_epoch = 0
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Initialize best model tracking
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        best_model_state = None
        best_epoch = -1
        epochs_without_improvement = 0
        
        # Log hyperparameters
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("device", device)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_function", "MSELoss")
        mlflow.log_param("train_dataset_size", len(train_loader.dataset))
        if val_loader:
            mlflow.log_param("val_dataset_size", len(val_loader.dataset))
        mlflow.log_param("random_seed", seed)
        
        if already_trained and checkpoint_path is None:
            raise Exception("Checkpoint path has to be specified if already_trained is True")
        
        if already_trained and checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            history = checkpoint['history']
            start_epoch = checkpoint['epoch'] + 1
            # Load best model tracking info if available
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_train_loss = checkpoint.get('best_train_loss', float('inf'))
            best_epoch = checkpoint.get('best_epoch', -1)
            mlflow.log_param("resumed_from_epoch", start_epoch)
            mlflow.log_param("resumed_best_val_loss", best_val_loss)
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx, (pred_data, imerg_data) in enumerate(pbar):
                    # Move data to device
                    pred_data = pred_data.to(device)  # Shape: [batch_size, 2, 1, height, width]
                    imerg_data = imerg_data.to(device)  # Shape: [batch_size, 2, height, width]
                    
                    # Add channel dimension to IMERG data if needed
                    if len(imerg_data.shape) == 4:
                        imerg_data = imerg_data.unsqueeze(2)  # [batch_size, 2, 1, height, width]
                    
                    # Get target (second timestep of IMERG data)
                    target = imerg_data[:, 1]
                    
                    # Forward pass
                    optimizer.zero_grad()
                    refined_pred = model(pred_data)
                    
                    # Calculate loss (MSE between refined prediction and IMERG ground truth)
                    loss = criterion(refined_pred, target)
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    train_loss += loss.item()
                    pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                    
                    # Log batch-level information every 10 batches
                    if batch_idx % 10 == 0:
                        logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Log training loss to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            
            # Validation
            if val_loader is not None:
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for pred_data, imerg_data in val_loader:
                        # Move data to device
                        pred_data = pred_data.to(device)
                        imerg_data = imerg_data.to(device)
                        
                        # Add channel dimension to IMERG data if needed
                        if len(imerg_data.shape) == 4:
                            imerg_data = imerg_data.unsqueeze(2)
                        
                        # Get target (second timestep of IMERG data)
                        target = imerg_data[:, 1]
                        
                        # Forward pass
                        refined_pred = model(pred_data)
                        
                        # Calculate loss
                        loss = criterion(refined_pred, target)
                        val_loss += loss.item()
                
                # Calculate average validation loss
                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                
                # Log validation loss to MLflow
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                
                # Check if this is the best model so far
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                    
                    # Log best model metrics
                    mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                    mlflow.log_metric("best_epoch", best_epoch, step=epoch)
                    
                    # Create best checkpoint dictionary
                    best_checkpoint = {
                        'epoch': epoch,
                        'model': best_model_state,
                        'optimizer': optimizer.state_dict(),
                        'history': history,
                        'best_val_loss': best_val_loss,
                        'best_train_loss': avg_train_loss,
                        'best_epoch': best_epoch
                    }
                    # Save best model checkpoint using torch.save
                    temp_filename = f"best_model_epoch_{epoch+1}.pth"
                    with tempfile.NamedTemporaryFile(suffix='.pth', prefix=temp_filename, delete=False) as tmp:
                        torch.save(best_checkpoint, tmp.name)
                        # Log to MLflow
                        mlflow.log_artifact(tmp.name, f"best_model_checkpoint")
                    # Clean up temporary file
                    os.unlink(tmp.name)
                    logger.info(f"New best model saved at epoch {epoch+1} with val_loss: {avg_val_loss:.6f}")
                else:
                    epochs_without_improvement += 1
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch+1})")
            else:
                # If no validation loader, use training loss for best model selection
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                    
                    # Log best model metrics
                    mlflow.log_metric("best_train_loss", best_train_loss, step=epoch)
                    mlflow.log_metric("best_epoch", best_epoch, step=epoch)
                    
                    # Create best checkpoint dictionary
                    best_checkpoint = {
                        'epoch': epoch,
                        'model': best_model_state,
                        'optimizer': optimizer.state_dict(),
                        'history': history,
                        'best_train_loss': best_train_loss,
                        'best_epoch': best_epoch
                    }
                    temp_filename = f"best_model_epoch_{epoch+1}.pth"
                    with tempfile.NamedTemporaryFile(suffix='.pth', prefix=temp_filename, delete=False) as tmp:
                        torch.save(best_checkpoint, tmp.name)
                        # Log to MLflow
                        mlflow.log_artifact(tmp.name, f"best_model_checkpoint")
                    # Clean up temporary file
                    os.unlink(tmp.name)
                    logger.info(f"New best model saved at epoch {epoch+1} with train_loss: {avg_train_loss:.6f}")
                else:
                    epochs_without_improvement += 1
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Best Train Loss: {best_train_loss:.6f} (Epoch {best_epoch+1})")
            
            # Log checkpoint every 5 epochs
            if epoch % 5 == 0:
                checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'history': history,
                    'best_val_loss': best_val_loss if val_loader else best_train_loss,
                    'best_train_loss': best_train_loss,
                    'best_epoch': best_epoch
                }
                # Save checkpoint using torch.save
                temp_filename = f"checkpoint_epoch_{epoch+1}.pth"
                with tempfile.NamedTemporaryFile(suffix='.pth', prefix=temp_filename, delete=False) as tmp:
                    torch.save(checkpoint, tmp.name)
                    # Log to MLflow
                    mlflow.log_artifact(tmp.name, f"checkpoints")
                # Clean up temporary file
                os.unlink(tmp.name)
        
        with torch.inference_mode():
            input_example_save=train_loader.dataset[0][0].unsqueeze(0).cpu()
        # Log final model (current state)
        mlflow.pytorch.log_model(model, f"final_model",input_example=input_example_save)
        
        # Load and log best model
        if best_model_state is not None:
            # Create a new model instance and load best weights
            best_model = copy.deepcopy(model)  # Create new instance of same model class
            best_model.load_state_dict(best_model_state)
            best_model=best_model.cpu()
            mlflow.pytorch.log_model(best_model, "best_model", input_example=input_example_save)
            logger.info(f"Best model logged with input example shape: {input_example_save.shape}")
        else:
            logger.warning("No best model state found to log")
        
        # Log final metrics
        mlflow.log_metric("final_train_loss", history['train_loss'][-1])
        if val_loader:
            mlflow.log_metric("final_val_loss", history['val_loss'][-1])
            mlflow.log_metric("best_val_loss_final", best_val_loss)
        else:
            mlflow.log_metric("best_train_loss_final", best_train_loss)
        
        mlflow.log_metric("best_epoch_final", best_epoch)
        mlflow.log_metric("epochs_without_improvement", epochs_without_improvement)
        
        # Log training history directly to MLflow
        mlflow.log_dict(history, "training_artifacts/training_history.json")
        mlflow.log_artifact('/scratch/IITB/monsoon_lab/24d1236/pratham/Model/model_training.log',"Logs")
    
    return model, optimizer, history, num_epochs
