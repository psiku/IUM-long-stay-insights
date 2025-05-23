import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import numpy as np

# training
def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
            if len(batch) == 3:
                x_cat, x_num, labels = batch
                x_cat, x_num, labels = x_cat.to(device), x_num.to(device), labels.to(device)
                outputs = model(x_cat, x_num)
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x_cat, x_num, labels = batch
                    x_cat, x_num, labels = x_cat.to(device), x_num.to(device), labels.to(device)
                    outputs = model(x_cat, x_num)
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print("Training complete!")


# evaluating
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            if len(batch) == 3:
                x_cat, x_num, labels = batch
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                labels = labels.to(device)
                outputs = model(x_cat, x_num)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

            probs = F.softmax(outputs, dim=1)
            all_preds.append(probs)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    accuracy = accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    roc_auc = roc_auc_score(all_labels, all_preds[:, 1])
    avg_precision = average_precision_score(all_labels, all_preds[:, 1])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")

    return all_preds, all_labels
