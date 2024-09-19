import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

def evaluate_model(model, loader, device):
    model.eval()
    predictions, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    accuracy = accuracy_score(labels_list, predictions)
    f1 = f1_score(labels_list, predictions, average='macro')
    tn, fp, fn, tp = confusion_matrix(labels_list, predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, f1, sensitivity, specificity

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config):
    best_val_accuracy = 0
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_accuracy, train_f1, train_sensitivity, train_specificity = evaluate_model(model, train_loader, device)
        val_accuracy, val_f1, val_sensitivity, val_specificity = evaluate_model(model, val_loader, device)

        scheduler.step(running_loss / len(train_loader))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model
