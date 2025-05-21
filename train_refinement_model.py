import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_refinement_model(model, train_loader, val_loader=None, 
                          num_epochs=30, learning_rate=0.001,checkpoint_save_dir=None, 
                          device='cuda' if torch.cuda.is_available() else 'cpu',already_trained=False,checkpoint_path=None):
    """
    Training function for the precipitation refinement model
    
    Args:
        model: PrecipitationRefinementModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run training on ('cuda' or 'cpu')
        already_trained: Bool to say if model has already been trained
        checkpoint_path: Path to the pth file which will have history, optimizer state, number of epochs trained and model state_dict.
    Returns:
        model,optimizer,history,num_epochs
    """
    print(f'Using device: {device}')
    model = model.to(device)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    start_epoch=0
    history = {
        'train_loss': [],
        'val_loss': []
    }
    if checkpoint_save_dir is None:
        print("No checkpoint saving")
    if already_trained and checkpoint_path is None:
        raise Exception("Checkpoint path has to be specified if already_trained is True")
    if already_trained and checkpoint_path is not None:
        checkpoint=torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history=checkpoint['history']
        start_epoch=checkpoint['epoch']+1
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch,num_epochs):
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
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
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
            
            # Update learning rate scheduler
            # scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")
        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history 
            # 'lr_sched': lr_sched
        }
        parent_dir=os.path.dirname(checkpoint_save_dir)
        if epoch%20==0:
            save_dir=f"{parent_dir}/checkpoint{epoch}.pth"
            torch.save(checkpoint,save_dir)

    return model,optimizer,history,num_epochs