import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(20240608)
torch.manual_seed(20240608)


def input_data(batch_size, data):
    input_array = np.zeros([batch_size, 16, 4, 4])  # PyTorch expects channels first
    label_array = np.zeros([batch_size, 4])

    for i in range(batch_size):
        str0 = data.readline().strip()
        elements = str0.split(' ')

        for j in range(16):
            channel = int(elements[j + 1])
            input_array[i][channel][j // 4][j % 4] = 1  # One-hot encode the input data

        label_array[i][int(elements[18])] = 1  # One-hot encode the label

    return torch.tensor(input_array, dtype=torch.float32), torch.tensor(label_array, dtype=torch.float32)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(16, 222, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(222, 222, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(222, 222, kernel_size=2, padding=1)
        self.conv4 = nn.Conv2d(222, 222, kernel_size=2, padding=1)
        self.conv5 = nn.Conv2d(222, 222, kernel_size=2, padding=1)
        self.fc = nn.Linear(9 * 9 * 222, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = torch.softmax(self.fc(x), dim=1)
        return x


def train_model(model, criterion, optimizer, num_epochs, batch_size, data_size, data_path, log_file):
    # Log training progress
    learn_log = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ", Start training\n"
    with open(log_file, "a") as f:
        f.write(learn_log)

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        epoch_log = f"Epoch {epoch + 1}/{num_epochs}, Start time: {epoch_start_time}\n"
        print(epoch_log)
        with open(log_file, "a") as f:
            f.write(epoch_log)

        if epoch < 10:
            file_name = data_path + "shuffle_0" + str(epoch) + "a" + ".txt"
        else:
            file_name = data_path + "shuffle_" + str(epoch) + "a" + ".txt"

        data = open(file_name, 'r')
        num_processed = 0

        while num_processed < data_size:
            batch_start_time = datetime.now()

            # Get input data and labels
            image, label = input_data(batch_size, data)

            # Convert labels to class indices for PyTorch
            label_indices = torch.argmax(label, dim=1)

            # Train the model
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label_indices)
            loss.backward()
            optimizer.step()
            num_processed += batch_size

            if num_processed % 1000000 == 0:
                batch_end_time = datetime.now()
                time_elapsed = batch_end_time - batch_start_time
                batch_log = f"Epoch {epoch + 1}, Processed {num_processed}/{data_size} samples, Batch time: {time_elapsed}\n"
                print(batch_log)
                with open(log_file, "a") as f:
                    f.write(batch_log)

                with torch.no_grad():
                    # Calculate loss and accuracy
                    output = model(image)
                    loss_val = criterion(output, label_indices).item()
                    pred = torch.argmax(output, dim=1)
                    correct = (pred == label_indices).float().sum()
                    acc_val = correct / batch_size

                    log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Epoch {epoch + 1}, Processed {num_processed} samples, Loss: {loss_val}, Accuracy: {acc_val}\n"
                    with open(log_file, "a") as f:
                        f.write(log_entry)

                    # Save model
                    if num_processed == data_size:
                        torch.save(model.state_dict(), f'./c5_epoch_{epoch}_step_{num_processed}.pt')

        data.close()
        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_log = f"Epoch {epoch + 1} completed, Duration: {epoch_duration}\n"
        print(epoch_log)
        with open(log_file, "a") as f:
            f.write(epoch_log)


# Initialize the network, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Example usage
train_model(model, criterion, optimizer, num_epochs=60, batch_size=1000, data_size=10000000,
            data_path='//TRdata/', log_file='log_cnn_5.txt')
