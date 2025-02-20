from scipy.io import loadmat
import torch
import torch.optim as optim
import torch.nn as nn

import fno_2d

import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
data = loadmat('data.mat')

# Input tensor (batch_size, x, y, in_channels)
tensor_dcbardx = torch.tensor(data['dcbardx1_cases_all'], dtype=torch.float32).permute(2, 0, 1)
tensor_dcbardy = torch.tensor(data['dcbardx2_cases_all'], dtype=torch.float32).permute(2, 0, 1)
inputs = torch.stack((tensor_dcbardx, tensor_dcbardy), dim=-1)

# Output tensor (batch_size, x, y, out_channels)
tensor_upcpbar = torch.tensor(data['upcp_bar_cases_all'], dtype=torch.float32).permute(2, 0, 1)
tensor_vpcpbar = torch.tensor(data['vpcp_bar_cases_all'], dtype=torch.float32).permute(2, 0, 1)
outputs = torch.stack((tensor_upcpbar, tensor_vpcpbar), dim=-1)

# Train / test sets
train_size = 1500
test_size = 200

train_inputs, train_outputs = inputs[:train_size, :, :, :].to(device), outputs[:train_size, :, :, :].to(device)
test_inputs, test_outputs = inputs[-test_size:, :, :, :].to(device), outputs[-test_size:, :, :, :].to(device)

# Data dimensions
print(f"Train Input Shape: {train_inputs.shape}, Train Output Shape: {train_outputs.shape}")
print(f"Test Input Shape: {test_inputs.shape}, Test Output Shape: {test_outputs.shape}")

# Model parameters
in_channels = 4
out_channels = 2
mid_channels = 20
modes_x = 12
modes_y = 12

# Setting up model
model = fno_2d.FNO2d(in_channels, out_channels, mid_channels, modes_x, modes_y).to(device)

# Learning parameters
epochs = 100
learning_rate = 0.001
regularization = 0.0001
mini_batch = train_size // epochs

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
loss_function = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    curr = time.time()

    # Mini-batch
    train_inputs_mini = train_inputs[epoch * mini_batch:(epoch + 1) * mini_batch, :, :, :]
    train_outputs_mini = train_outputs[epoch * mini_batch:(epoch + 1) * mini_batch, :, :, :]

    # Forward pass
    predictions = model(train_inputs_mini)

    # Compute loss
    loss = loss_function(predictions.squeeze(), train_outputs_mini.squeeze())

    # Backward pass
    loss.backward()
    optimizer.step()

    # Update info
    print(f"Epoch [{epoch + 1} / {epochs}], Loss: {loss.item() * 100:.6f}%, Computing time: {time.time() - curr:.6f}s")

# Testing model
model.eval()
with torch.no_grad():
    predictions = model(test_inputs)
    loss_test = loss_function(predictions.squeeze(), test_outputs.squeeze())
print(f"Test Loss: {loss_test.item() * 100:.6f}%")
