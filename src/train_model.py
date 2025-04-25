import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix

def setup_output(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.json")
    cm_path = os.path.join(output_dir, "confusion_matrix.json")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            json.dump([], f)
    if not os.path.exists(cm_path):
        with open(cm_path, "w") as f:
            json.dump({}, f)
    return log_path, cm_path

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x).logits
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        p = logits.argmax(dim=1)
        preds.extend(p.cpu().tolist())
        labels.extend(y.cpu().tolist())
    loss = running_loss / len(loader.dataset)
    f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds).tolist()
    return loss, f1, cm

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).logits
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            p = logits.argmax(dim=1)
            preds.extend(p.cpu().tolist())
            labels.extend(y.cpu().tolist())
            correct += (p == y).sum().item()
            total += y.size(0)
    loss = running_loss / len(loader.dataset)
    acc = correct / total
    f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds).tolist()
    return loss, acc, f1, cm

def log_metrics(log_path, cm_path, epoch, train_metrics, val_metrics):
    train_loss, train_f1, train_cm = train_metrics
    val_loss, val_acc, val_f1, val_cm = val_metrics
    with open(log_path, "r+") as f:
        logs = json.load(f)
        logs.append({"epoch": epoch, "train_loss": train_loss, "train_f1": train_f1,
                     "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})
        f.seek(0); json.dump(logs, f, indent=2); f.truncate()
    with open(cm_path, "r+") as f:
        cms = json.load(f)
        cms[str(epoch)] = {"train": train_cm, "val": val_cm}
        f.seek(0); json.dump(cms, f, indent=2); f.truncate()

def save_checkpoint(model, output_dir, model_name, epoch):
    path = os.path.join(output_dir, f"{model_name}_epoch{epoch}.pth")
    torch.save(model.state_dict(), path)
    return path

def train_model(
    model, train_loader, val_loader,
    epochs=10, lr=3e-4, device=None,
    output_dir="./saved_models", patience=3
):
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    log_path, cm_path = setup_output(output_dir)
    model_name = type(model).__name__.lower()
    best_val = float('inf')
    no_imp = 0
    
    for epoch in range(1, epochs+1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        log_metrics(log_path, cm_path, epoch, train_metrics, val_metrics)
        cp = save_checkpoint(model, output_dir, model_name, epoch)
        print(f"Epoch {epoch}/{epochs} - Train loss: {train_metrics[0]:.4f}, Train F1: {train_metrics[1]:.4f} "
              f"Val loss: {val_metrics[0]:.4f}, Val Acc: {val_metrics[1]:.4f}, Val F1: {val_metrics[2]:.4f}")
        if val_metrics[0] < best_val:
            best_val = val_metrics[0]; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
    return model
