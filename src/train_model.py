import torch
import torch.nn as nn
import torch.optim as optim

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 10,
    lr: float = 3e-4,
    device: str = None
) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).logits
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch}: train loss {running_loss/len(train_loader.dataset):.4f}")

        model.eval()
        val_loss = correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).logits
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"Valid loss: {val_loss/len(val_loader.dataset):.4f}\nValid acc: {correct/total:.4f}\n")

    return model