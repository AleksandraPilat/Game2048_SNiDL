import torch
import torch.nn as nn
import torch.optim as optim

import training


# Define a Convolutional Neural Network (CNN) class
class CNN5Sym(nn.Module):
    def __init__(self):
        super(CNN5Sym, self).__init__()
        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(16, 222, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(222, 222, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(222, 222, kernel_size=2, padding=1)
        self.conv4 = nn.Conv2d(222, 222, kernel_size=2, padding=1)
        self.conv5 = nn.Conv2d(222, 222, kernel_size=2, padding=1)
        # Define the fully connected layer
        self.fc = nn.Linear(9 * 9 * 222, 4)

    def forward(self, x):
        # Forward pass through the convolutional layers with ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        # Flatten the output from the last convolutional layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # Initialize the network and move it to the appropriate device
    model = CNN5Sym().to(training.device)

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
        model_name="cnn5_sym",
        log_file='log_cnn5_sym.txt'
    )
