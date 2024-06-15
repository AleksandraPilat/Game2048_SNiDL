import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(20240608)
torch.manual_seed(20240608)

# Determine the device to use for computation (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


def load_batch_data(batch_size, data):
    """
    Load a batch of input data and labels from the given data file.

    Returns:
        tuple: Tensors containing the input data and labels.
    """
    input_array = np.zeros([batch_size, 16, 4, 4])
    label_array = np.zeros([batch_size, 4])

    for i in range(batch_size):
        line = data.readline().strip()
        elements = line.split(' ')

        for j in range(16):
            channel = int(elements[j + 1])
            input_array[i][channel][j // 4][j % 4] = 1  # One-hot encode the input data

        label_array[i][int(elements[18])] = 1  # One-hot encode the label

    return torch.tensor(input_array, dtype=torch.float32).to(device), torch.tensor(label_array, dtype=torch.float32).to(
        device)


def train_model(model, criterion, optimizer, num_epochs, batch_size, data_size, data_path, model_name, log_file):
    """
    Train the given model using the provided training parameters.
    """
    # Log the start of training
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

        # Determine the data file to read based on the epoch

        root_dir = Path(__file__).parent.parent
        file_name = root_dir / data_path / f"shuffle_{epoch:02d}a.txt"
        data = open(file_name, 'r')
        num_processed = 0

        while num_processed < data_size:
            batch_start_time = datetime.now()

            # Load input data and labels
            image, label = load_batch_data(batch_size, data)

            # Convert labels to class indices for PyTorch
            label_indices = torch.argmax(label, dim=1)

            # Train the model
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label_indices)
            loss.backward()
            optimizer.step()
            num_processed += batch_size

            if num_processed % data_size == 0:
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
                        model_dir = root_dir / f'{model_name}_models'
                        model_dir.mkdir(exist_ok=True)
                        torch.save(model.state_dict(), model_dir / f'{model_name}_{epoch}_step_{num_processed}.pt')

        data.close()
        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_log = f"Epoch {epoch + 1} completed, Duration: {epoch_duration}\n"
        print(epoch_log)
        with open(log_file, "a") as f:
            f.write(epoch_log)