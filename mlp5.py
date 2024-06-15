import torch.nn as nn
import torch.optim as optim

import training


class MLP5(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4 * 4 * 16, 512)
        self.layer2 = nn.Linear(512, 1024)
        self.layer3 = nn.Linear(1024, 2048)
        self.layer4 = nn.Linear(2048, 512)
        self.layer5 = nn.Linear(512, 4)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        x = nn.ReLU()(x)
        x = self.layer3(x)
        x = nn.ReLU()(x)
        x = self.layer4(x)
        x = nn.ReLU()(x)
        x = self.layer5(x)
        return x


if __name__ == '__main__':
    # Initialize the network and move it to the appropriate device
    model = MLP5().to(training.device)

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
        model_name="mlp5",
        log_file='log_mlp5.txt'
    )
