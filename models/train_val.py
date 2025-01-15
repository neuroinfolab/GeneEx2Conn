# GeneEx2Conn/models/train_val.py

from imports import *

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, verbose=True):
    train_history = {"train_loss": [], "val_loss": [], "train_pearson": [], "val_pearson": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        train_history["train_loss"].append(train_metrics["loss"])
        train_history["train_pearson"].append(train_metrics["pearson"])
        
        if val_loader:
            val_metrics = evaluate(model, val_loader, criterion, device)
            train_history["val_loss"].append(val_metrics["loss"])
            train_history["val_pearson"].append(val_metrics["pearson"])
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            
        elif verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_metrics['loss']:.4f}")

    return train_history


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_train_loss = 0
    train_pearson_values = []
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X).squeeze()
        try:
            loss = criterion(predictions, batch_y)
        except:
            loss = criterion(predictions, batch_y, model)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        pearson = PearsonCorrCoef().to(device)    
        train_pearson_values.append(pearson(predictions, batch_y).item())
    
    return {
        "loss": total_train_loss / len(train_loader),
        "pearson": np.mean(train_pearson_values)
    }


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    val_pearson_values = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X).squeeze()
            try: 
                val_loss = criterion(predictions, batch_y)
            except:
                val_loss = criterion(predictions, batch_y, model)
            total_val_loss += val_loss.item()
            pearson = PearsonCorrCoef().to(device)
            val_pearson_values.append(pearson(predictions, batch_y).item())
    
    return {
        "loss": total_val_loss / len(val_loader),
        "pearson": np.mean(val_pearson_values)
    }