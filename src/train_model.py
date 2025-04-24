import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 10,
    lr: float = 3e-4,
    device: str = None,
    output_dir: str = "./saved_models"
) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.json")
    cm_path  = os.path.join(output_dir, "confusion_matrix.json")
    model_name = type(model).__name__.lower()

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            json.dump([], f)
    if not os.path.exists(cm_path):
        with open(cm_path, "w") as f:
            json.dump({}, f)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).logits
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(y.cpu().tolist())

        train_loss = running_loss / len(train_loader.dataset)
        train_f1   = f1_score(train_labels, train_preds, average="macro")
        train_cm   = confusion_matrix(train_labels, train_preds).tolist()

        model.eval()
        running_val_loss = correct = total = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).logits
                loss = criterion(logits, y)
                running_val_loss += loss.item() * x.size(0)

                preds = logits.argmax(dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(y.cpu().tolist())
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc  = correct / total
        val_f1   = f1_score(val_labels, val_preds, average="macro")
        val_cm   = confusion_matrix(val_labels, val_preds).tolist()

        with open(log_path, "r+") as f:
            logs = json.load(f)
            logs.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1
            })
            f.seek(0)
            json.dump(logs, f, indent=2)
            f.truncate()

        with open(cm_path, "r+") as f:
            cms = json.load(f)
            cms[str(epoch)] = {
                "train": train_cm,
                "val": val_cm
            }
            f.seek(0)
            json.dump(cms, f, indent=2)
            f.truncate()

        model_path = os.path.join(output_dir, f"{model_name}_epoch{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch}/{epochs} - Train loss: {train_loss:.4f}, Train F1: {train_f1:.4f} \n Val loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    return model
