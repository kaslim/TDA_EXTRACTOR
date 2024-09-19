import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model.model import CNN
from model.train import train_model, evaluate_model
from model.data_loader import prepare_data
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.000005,
    "random_state": 43,
    "early_stopping_patience": 15
}

train_file_paths = ['training-a_features.npy', 'training-b_features.npy', 'training-c_features.npy','training-d_features.npy'
                    'training-e_features.npy','training-f_features.npy']
val_file_paths = ['validation_features.npy']
X_train, X_val, y_train, y_val = prepare_data(train_file_paths, val_file_paths, config)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

model = CNN(input_length=X_train.shape[1]).to(device)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config)

# Final evaluation
val_accuracy, val_f1, val_sensitivity, val_specificity = evaluate_model(trained_model, val_loader, device)
print(f'Validation Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}, Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}')

# Save the final model
torch.save(trained_model.state_dict(), f"trained_model_{val_accuracy:.4f}.pth")
