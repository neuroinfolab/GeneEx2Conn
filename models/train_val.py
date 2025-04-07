from env.imports import *
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, patience=100, scheduler=None, verbose=True):
    train_history = {"train_loss": [], "val_loss": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Memory Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")
    
    cudnn.benchmark = True  # Auto-tune GPU kernels
    scaler = GradScaler()  # Enable FP16 training
    scaler = None

    best_val_loss = float("inf")  # Track the best validation loss
    best_model_state = None  # Store the best model state
    patience_counter = 0  # Counts epochs without improvement

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)

        train_history["train_loss"].append(train_loss)

        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device, scheduler)
            train_history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Best val loss so far at epoch {epoch+1}: {best_val_loss:.4f}")
                best_model_state = model.state_dict()  # Save best model
                patience_counter = 0  # Reset counter if improvement
            else:
                patience_counter += 1  # Increment counter if no improvement

            # Early stopping if patience threshold reached
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}. Restoring best model with Val Loss: {best_val_loss:.4f}")
                model.load_state_dict(best_model_state)  # Rewind to best model
                break
            elif epoch == epochs - 1: # always rewind to best model at end of training
                model.load_state_dict(best_model_state)  # Rewind to best model
                print("Model state dict loaded from best model")
                
        elif verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    return train_history

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Combined training function that handles both regular and mixed precision training"""
    model.train()
    total_train_loss = 0
    
    for batch_X, batch_y, batch_coords, batch_idx in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_coords = batch_coords.to(device)
        
        optimizer.zero_grad()

        if scaler is not None: # Mixed precision training path            
            with autocast():
                if hasattr(model, 'include_coords'):
                    predictions = model(batch_X, batch_coords).squeeze()
                else:
                    predictions = model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: # Regular training path
            if hasattr(model, 'include_coords'):
                predictions = model(batch_X, batch_coords).squeeze()
            else:
                predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
        
        total_train_loss += loss.item()
    
    return total_train_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device, scheduler=None):
    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_X, batch_y, batch_coords, batch_idx in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_coords = batch_coords.to(device)
            

            if hasattr(model, 'include_coords'):
                predictions = model(batch_X, batch_coords).squeeze()
            else:
                predictions = model(batch_X).squeeze()
                
            val_loss = criterion(predictions, batch_y)            
            total_val_loss += val_loss.item()
            
            # integrate these eventually 
            if hasattr(model, 'binarize') and model.binarize:
                pred_labels = (torch.sigmoid(predictions) > 0.5).float()
            else:
                pred_labels = predictions.round()
            
            total_correct += (pred_labels == batch_y).sum().item()
            total_samples += batch_y.size(0)
    
    mean_val_loss = total_val_loss / len(val_loader)
    accuracy = total_correct / total_samples

    if scheduler is not None:
        prev_lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step(mean_val_loss)
        new_lr = scheduler.optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"\nLR REDUCED: {prev_lr:.6f} → {new_lr:.6f} at Val Loss: {mean_val_loss:.6f}")

    return mean_val_loss