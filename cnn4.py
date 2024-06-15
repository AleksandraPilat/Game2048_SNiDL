import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import training


# Define a Convolutional Neural Network (CNN) class
class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()
        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=256, kernel_size=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.relu4 = nn.ReLU()
        self.flatten = nn.Flatten()
        # Define a fully connected layer
        self.fc1 = nn.Linear(256 * 4 * 4, 4)

    def forward(self, x):
        # Forward pass through the network with padding applied before each convolution
        x = F.pad(x, (1, 0, 1, 0))  # Pad (right, left, bottom, top)
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.pad(x, (1, 0, 1, 0))  # Pad (right, left, bottom, top)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.pad(x, (1, 0, 1, 0))  # Pad (right, left, bottom, top)
        x = self.conv3(x)
        x = self.relu3(x)
        x = F.pad(x, (1, 0, 1, 0))  # Pad (right, left, bottom, top)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    # Initialize the network and move it to the appropriate device
    model = CNN4().to(training.device)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Example usage: Train the model
    training.train_model(
        model,
        criterion,
        optimizer,
        num_epochs=60,
        batch_size=1_000,
        data_size=10_000_000,
        data_path='TRdata/',
        model_name="cnn4",
        log_file='log_cnn4.txt'
    )
