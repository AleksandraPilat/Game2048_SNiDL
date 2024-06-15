import torch.nn as nn
import torch.optim as optim

import training


# Define a Multi-Layer Perceptron (MLP) class
class MLP4(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the fully connected layers
        self.layer1 = nn.Linear(4 * 4 * 16, 356)
        self.layer2 = nn.Linear(356, 1024)
        self.layer3 = nn.Linear(1024, 356)
        self.layer4 = nn.Linear(356, 4)

    def forward(self, x):
        # Flatten the input tensor
        x = x.reshape(x.size(0), -1)
        # Forward pass through the fully connected layers with ReLU activation
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        x = nn.ReLU()(x)
        x = self.layer3(x)
        x = nn.ReLU()(x)
        x = self.layer4(x)
        return x


if __name__ == '__main__':
    # Initialize the network and move it to the appropriate device
    model = MLP4().to(training.device)

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
        model_name="mlp4",
        log_file='log_mlp4.txt'
    )
