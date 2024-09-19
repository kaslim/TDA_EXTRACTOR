import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_length):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.4)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm1d(128)

        conv_output_length = input_length
        for _ in range(4):
            conv_output_length = (conv_output_length - 2) // 2

        self.fc1 = nn.Linear(128 * conv_output_length, 64)
        self.drop5 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool1(self.bn1(torch.relu(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))
        x = self.drop2(x)
        x = self.pool3(self.bn3(torch.relu(self.conv3(x))))
        x = self.pool4(self.bn4(torch.relu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.drop5(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
