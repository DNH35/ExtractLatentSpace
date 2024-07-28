import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import random

# Create a 1-D environment with Gaussian filter
def create_environment(length, sigma):
    environment = np.random.rand(length)
    smoothed_environment = gaussian_filter1d(environment, sigma=sigma)
    return smoothed_environment

# Simulate agent's movement
def simulate_agent(environment, timesteps):
    """Simulate an agent moving through the 1-D environment."""
    # Start at the center of the environment
    position = len(environment) // 2
    positions = [position]

    for _ in range(timesteps):
        # Simple movement logic: move randomly left or right
        left = random.randint(-len(environment) // 2, 0)
        right = random.randint(0, len(environment) // 2)
        move = random.randint(left, right)
        
        position = max(0, min(position + move, len(environment) - 1))
        positions.append(position)

    return positions

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, input_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(input_dim, 1)
        self.activations = []

    def forward(self, x):
        self.activations = []
        x = self.embedding(x)

        for layer in self.transformer_layers:
            x = layer(x)
            self.activations.append(x.detach().cpu().numpy())  # Store activations

        return self.fc_out(x)

# Define the Dataset
class PositionDataset(Dataset):
    def __init__(self, positions, seq_length):
        self.positions = positions
        self.seq_length = seq_length

    def __len__(self):
        return len(self.positions) - self.seq_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.positions[index:index + self.seq_length], dtype=torch.long),
            torch.tensor(self.positions[index + 1:index + self.seq_length + 1], dtype=torch.float)
        )

# Parameters
environment_length = 10
sigma = 5
timesteps = 100
seq_length = 10
batch_size = 4
input_dim = environment_length
num_heads = 2
num_layers = 2
learning_rate = 0.001
num_epochs = 20

def plot_activations(activations, layer_num):
    """Plot the average activations for a given layer number."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(activations, cmap='viridis')
    plt.title(f'Average Neuron Activations for Layer {layer_num}')
    plt.xlabel('Sequence Position')
    plt.ylabel('Neuron')
    plt.show()


# Create the environment and simulate agent
environment = create_environment(environment_length, sigma)
positions = simulate_agent(environment, timesteps)

# Create dataset and data loader
dataset = PositionDataset(positions, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, criterion, and optimizer
model = TransformerModel(input_dim, num_heads, num_layers, seq_length)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        x, y = batch
        optimizer.zero_grad()
        output = model(x).squeeze(-1)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

average_activations = []

for layer_activations in model.activations:
    # Convert to numpy array and calculate mean over the 0th axis (sequence positions)
    average_activations.append(np.mean(np.array(layer_activations), axis=0))

# Plot average activations for each layer
for i, activations in enumerate(average_activations):
    # Reshape activations to (num_neurons, seq_length)
    activations_reshaped = activations.T  # Transpose for plotting
    plot_activations(activations_reshaped, i + 1)

# Evaluate and capture activations
model.eval()

# Plot the results
predicted_positions = []
with torch.no_grad():
    for i in range(len(positions) - seq_length):
        input_seq = torch.tensor(positions[i:i + seq_length], dtype=torch.long).unsqueeze(0)
        predicted_position = model(input_seq).squeeze(0).argmax().item()
        predicted_positions.append(predicted_position)

# Plot the environment
plt.figure(figsize=(12, 6))
plt.plot(environment, label='Environment')
plt.xlabel('Position')
plt.ylabel('Value')
plt.show()

# Plot the agent's path and predicted path
plt.figure(figsize=(12, 6))
plt.scatter(range(seq_length, len(positions)),
            [environment[pos] for pos in positions[seq_length:]],
            color='red', s=10, label='Agent Path')
plt.scatter(range(seq_length, len(predicted_positions) + seq_length),
            [environment[pos] for pos in predicted_positions],
            color='blue', label='Predicted Path')
plt.title('1-D Environment with Gaussian Filter and Predicted Path')
plt.xlabel('Time Step')
plt.ylabel('Environment Value')
plt.legend()
plt.show()
