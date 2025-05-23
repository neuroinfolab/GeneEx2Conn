from env.imports import *
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from data.data_utils import augment_batch


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, patience=100, scheduler=None, verbose=True, dataset=None):        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Memory Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")
    cudnn.benchmark = True  # Auto-tune GPU kernels
    scaler = GradScaler()  # Enable FP16 training for faster training - set to None for regular training
    
    train_history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf") 
    best_model_state = None  # Store the best model state
    patience_counter = 0  # Counts epochs without improvement
    
    for epoch in range(epochs):
        start_time = time.time() if (epoch + 1) % 5 == 0 else None    
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=scaler, dataset=dataset)
        train_history["train_loss"].append(train_loss)
        
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device, scheduler)
            train_history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Best val loss so far at epoch {epoch+1}: {best_val_loss:.4f}")
                best_model_state = model.state_dict()  # Save best model
                patience_counter = 0  # Reset counter if improvement
            else:
                patience_counter += 1  # Increment counter if no improvement

            if patience_counter >= patience or epoch == epochs - 1:
                model.load_state_dict(best_model_state)  # Rewind to best model
                predictions, targets = model.predict(val_loader)
                pearson_corr = pearsonr(predictions, targets)[0]
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}. Restoring best model with Val Loss: {best_val_loss:.4f}, Pearson Correlation: {pearson_corr:.4f}")
                else:
                    print(f"\nReached final epoch {epoch+1}. Restoring best model with Val Loss: {best_val_loss:.4f}, Pearson Correlation: {pearson_corr:.4f}")
                break
        
        if verbose and (epoch + 1) % 5 == 0:
            epoch_time = time.time() - start_time
            try: 
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
            except: 
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s")

    return train_history

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None, dataset=None):
    model.train()
    total_train_loss = 0

    for batch_X, batch_y, batch_coords, batch_idx in train_loader:
        if dataset is not None: # Target-side augmentation with given linear decaying Pr only for transformer models
            if np.random.random() < model.aug_prob*(1-epoch/model.epochs) : batch_y = augment_batch(batch_idx, dataset, device)
        
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_coords = batch_coords.to(device)
        
        optimizer.zero_grad()
        if scaler is not None: # Mixed precision training path            
            with autocast(dtype=torch.bfloat16):
                if hasattr(model, 'include_coords'): # For models with CLS
                    predictions = model(batch_X, batch_coords).squeeze()
                elif hasattr(model, 'optimize_encoder'):
                    predictions = model(batch_X, batch_idx).squeeze()
                else:
                    predictions = model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: # Regular training path
            if hasattr(model, 'include_coords'): # For models with CLS
                predictions = model(batch_X, batch_coords).squeeze()
            elif hasattr(model, 'optimize_encoder'): # For PLS models
                predictions = model(batch_X, batch_idx).squeeze()
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
            elif hasattr(model, 'optimize_encoder'):
                predictions = model(batch_X, batch_idx).squeeze()
            else:
                predictions = model(batch_X).squeeze()
            
            val_loss = criterion(predictions, batch_y)            
            total_val_loss += val_loss.item()
            
            if hasattr(model, 'binarize') and model.binarize: # integrate these eventually 
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