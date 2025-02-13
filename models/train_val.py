from env.imports import *

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler=None, verbose=True):
    train_history = {"train_loss": [], "val_loss": [], "train_pearson": [], "val_pearson": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        train_history["train_loss"].append(train_metrics["loss"])
        train_history["train_pearson"].append(train_metrics["pearson"])
        
        if val_loader:
            val_metrics = evaluate(model, val_loader, criterion, device, scheduler)
            train_history["val_loss"].append(val_metrics["loss"])
            train_history["val_pearson"].append(val_metrics["pearson"])
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                #print(f"Epoch {epoch+1}/{epochs}, Train Pearson: {train_metrics['pearson']:.4f}, Val Pearson: {val_metrics['pearson']:.4f}")

        elif verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_metrics['loss']:.4f}")

    return train_history

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_train_loss = 0
    train_pearson_values = []
    pearson = PearsonCorrCoef().to(device)
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X).squeeze()
        try: 
            loss = criterion(predictions, batch_y)
        except:
            loss = criterion(predictions, batch_y, model)

        total_train_loss += loss.item() # running sum of the loss over all batches
        train_pearson_values.append(pearson(predictions, batch_y).item()) # this is a list of pearson corrs for each batch in the epoch

        loss.backward()
        optimizer.step()
    
    mean_train_loss = total_train_loss / len(train_loader)
    mean_train_pearson = np.mean(train_pearson_values)
    
    return {
        "loss": mean_train_loss,
        "pearson": mean_train_pearson
    }

def evaluate(model, val_loader, criterion, device, scheduler=None):
    model.eval()
    total_val_loss = 0
    val_pearson_values = []
    pearson = PearsonCorrCoef().to(device)
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X).squeeze()
            try: 
                val_loss = criterion(predictions, batch_y)
            except:
                val_loss = criterion(predictions, batch_y, model)
            
            total_val_loss += val_loss.item()
            val_pearson_values.append(pearson(predictions, batch_y).item()) # this might not make sense to track...
    
    mean_val_loss = total_val_loss / len(val_loader)
    mean_val_pearson = np.mean(val_pearson_values)

    if scheduler is not None: # NEED TO GET THIS TO WORK 
       scheduler.step(mean_val_loss)

    return {
        "loss": mean_val_loss,
        "pearson": mean_val_pearson
    }