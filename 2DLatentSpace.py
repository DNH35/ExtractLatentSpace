import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

# Environment setup
grid_size = 64
environment = np.random.rand(grid_size, grid_size, 3)
environment = gaussian_filter(environment, sigma=1)  # Apply spatial autocorrelation

# Agent setup
sensor_angle = 90
num_sensors = 5

# Function to simulate agent's movement and sensor readings
def move_agent(pos, direction, step_size=1):
    angle = np.radians(direction)
    new_pos = pos + step_size * np.array([np.cos(angle), np.sin(angle)])
    new_pos = np.clip(new_pos, 0, grid_size - 1)
    return new_pos

def get_sensor_readings(pos, direction, env, num_sensors=5, sensor_angle=90):
    readings = []
    for i in range(num_sensors):
        angle = direction + (i - num_sensors // 2) * (sensor_angle / num_sensors)
        angle_rad = np.radians(angle)
        sensor_pos = pos + 10 * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        sensor_pos = np.clip(sensor_pos, 0, grid_size - 1).astype(int)
        readings.append(env[sensor_pos[0], sensor_pos[1]])
    return np.array(readings).flatten()


class PredictiveTransformer(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, output_size):
        super(PredictiveTransformer, self).__init__()
        self.input_layer = nn.Linear(input_size, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(dim_feedforward, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        x = self.output_layer(x)
        return x

# Model setup
input_size = num_sensors * 3  # Assuming each sensor reads RGB values
num_heads = 4
num_layers = 2
dim_feedforward = 100
output_size = input_size

model = PredictiveTransformer(input_size, num_heads, num_layers, dim_feedforward, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Simulate agent's exploration
num_steps = 10000
agent_pos = np.array([grid_size // 2, grid_size // 2])
agent_dir = 0  # Direction in degrees

positions = []
directions = []
readings_list = []

for step in range(num_steps):
    agent_dir += np.random.normal(0, 30)
    agent_pos = move_agent(agent_pos, agent_dir)
    readings = get_sensor_readings(agent_pos, agent_dir, environment)
    
    positions.append(agent_pos.copy())
    directions.append(agent_dir)
    readings_list.append(readings)
    
    if step % 100 == 0:
        print(f'Step [{step}/{num_steps}]')

# Train the PredictiveTransformer model using the collected data
for step in range(1, num_steps):
    input_data = torch.tensor(readings_list[step - 1].reshape(1, -1), dtype=torch.float32)
    target_data = torch.tensor(readings_list[step].reshape(1, -1), dtype=torch.float32)
    
    model.train()
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target_data)
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f'Training step [{step}/{num_steps}], Loss: {loss.item():.4f}')

# Collect activations after training
model.eval()
num_neurons = dim_feedforward
activations = np.zeros((grid_size, grid_size, num_neurons))

for pos, direction, readings in zip(positions, directions, readings_list):
    pos = pos.astype(int)
    with torch.no_grad():
        sensor_readings = torch.tensor(readings, dtype=torch.float32).unsqueeze(0)
        activities = model.transformer(model.input_layer(sensor_readings).unsqueeze(1)).squeeze(1).numpy().flatten() 
    activations[pos[0], pos[1], :len(activities)] += activities

# Rotate the activations by 90 degrees
activations = np.rot90(activations, k=1, axes=(0, 1))

# Calculate the average activation
average_activations = activations.mean(axis=(0, 1))
print("activations ", activations)

# Generate the heat map
fig, axes = plt.subplots(8, 8, figsize=(16, 16))  # Adjusting the layout to match the provided image
for i, ax in enumerate(axes.flatten()):
    sns.heatmap(activations[:, :, i % num_neurons], ax=ax, cbar=False, square=True, cmap='hot')
    ax.set_xticks([])
    ax.set_yticks([])

# Add a color bar to one of the subplots to act as a legend
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
sns.heatmap(activations[:, :, 0], ax=axes[0, 0], cbar_ax=cbar_ax, square=True, cmap='hot')

plt.suptitle('Latent Space Average Activation')
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rect to make space for the color bar
plt.show()

# Plot the environment
plt.figure(figsize=(8, 8))
plt.imshow(environment, extent=(0, grid_size, 0, grid_size))
plt.title('Environment')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Color Intensity')
plt.show()

# Plot the agent's path
positions_array = np.array(positions)
plt.figure(figsize=(8, 8))
plt.imshow(environment, extent=(0, grid_size, 0, grid_size))
plt.plot(positions_array[:, 0], positions_array[:, 1], color='red', linewidth=1)
plt.scatter(positions_array[0, 0], positions_array[0, 1], color='blue', label='Start', s=100)
plt.scatter(positions_array[-1, 0], positions_array[-1, 1], color='green', label='End', s=100)
plt.title('Agent Path in the Environment')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
