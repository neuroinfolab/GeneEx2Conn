from env.imports import *

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler=None, verbose=True):
    train_history = {"train_loss": [], "val_loss": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        train_history["train_loss"].append(train_loss)
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device, scheduler)
            train_history["val_loss"].append(val_loss)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        elif verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    return train_history


def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_train_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X).squeeze()
        loss = criterion(predictions, batch_y)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return total_train_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device, scheduler=None):
    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X).squeeze()
            val_loss = criterion(predictions, batch_y)            
            total_val_loss += val_loss.item()
            
            if hasattr(model, 'binarize') and model.binarize:
                pred_labels = (torch.sigmoid(predictions) > 0.5).float()
            else:
                pred_labels = predictions.round()
            
            total_correct += (pred_labels == batch_y).sum().item()
            total_samples += batch_y.size(0)
    
    mean_val_loss = total_val_loss / len(val_loader)
    print(f'Mean Val Loss: {mean_val_loss:.4f}')
    accuracy = total_correct / total_samples
    print(f'Validation Accuracy: {accuracy:.4f}')

    if scheduler is not None:
        prev_lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step(mean_val_loss)
        new_lr = scheduler.optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"\nLR REDUCED: {prev_lr:.6f} → {new_lr:.6f} at Val Loss: {mean_val_loss:.6f}")

    return mean_val_loss