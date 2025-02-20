import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import fno

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Generate synthetic Navier-Stokes dataset
def generate_nse_dataset(grid_size=16, time_steps=5, num_samples=50):
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    t = np.linspace(0, 1, time_steps)

    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    U = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.exp(-T)  # Synthetic velocity field
    U += 0.05 * np.random.randn(grid_size, grid_size, time_steps)  # Small perturbations
    print(U.shape)

    inputs, outputs = [], []
    for _ in range(num_samples):
        start_t = np.random.randint(0, time_steps - 2)
        input_sample = U[:, :, start_t:start_t + 2]  # Two time steps as input
        output_sample = U[:, :, start_t + 2]  # Predict next step

        inputs.append(input_sample)
        outputs.append(output_sample)

    # Convert to PyTorch tensors and reshape
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32).unsqueeze(-1)  # (batch, x, y, t, channel)
    outputs = torch.tensor(np.array(outputs), dtype=torch.float32).unsqueeze(-1)  # (batch, x, y, channel)

    return inputs.to(device), outputs.to(device)

# Generate dataset
train_inputs, train_outputs = generate_nse_dataset()

# # Display a sample
# plt.imshow(train_inputs[0, 0, :, :, -1], cmap="viridis")
# plt.title("Sample Initial Condition (Last Time Step)")
# plt.colorbar()
# plt.show()
#
# # Display ground truth
# plt.imshow(train_outputs[0, 0, :, :], cmap="inferno")
# plt.title("Sample Ground Truth (Next Time Step)")
# plt.colorbar()
# plt.show()

# Data dimensions
batch_size, channels, x_dim, y_dim, t_dim = train_inputs.shape
print(f"Train Input Shape: {train_inputs.shape}, Train Output Shape: {train_outputs.shape}")

# Move data to device
train_inputs, train_outputs = train_inputs.to(device), train_outputs.to(device)

# Define model parameters
in_channels = 1  # Velocity field
out_channels = 1  # Output velocity field
mid_channels = 8  # Intermediate channels
modes_x, modes_y, modes_t = 4, 4, 2  # Fourier modes

# Instantiate FNO3d model
model = fno.FNO3d(in_channels, out_channels, mid_channels, modes_x, modes_y, modes_t).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass
    predictions = model(train_inputs)[:, :, :, -1, :]

    # Compute loss
    print(predictions.shape)
    print(train_outputs.shape)
    loss = loss_fn(predictions.squeeze(), train_outputs.squeeze())

    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

# Test model on a single example
with torch.no_grad():
    test_input = train_inputs[0:1]
    test_output = train_outputs[0:1]
    prediction = model(test_input).cpu().numpy().squeeze()

# Visualize the results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(test_input.cpu().numpy().squeeze()[:, :, -1], cmap="viridis")
plt.title("Input (Last Time Step)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(test_output.cpu().numpy().squeeze(), cmap="inferno")
plt.title("Ground Truth (Next Time Step)")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(prediction[:, :, -1], cmap="magma")
plt.title("FNO3d Prediction")
plt.colorbar()

plt.show()

# Final loss
final_loss = loss.item()
print(f"Final Loss: {final_loss:.6f}")