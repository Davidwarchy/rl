# plot_statistics.py
import json
import matplotlib.pyplot as plt
import numpy as np

# Load data from JSON file
file_path = 'output/q_learning_data.json'  # Adjust the path accordingly
with open(file_path, 'r') as f:
    data = json.load(f)

q_table_history = data['q_table_history']
episode_iterations = np.log10(data['episode_iterations'])

# Calculate moving average
window_size = 10  # You can adjust this
moving_average = np.convolve(episode_iterations, np.ones(window_size) / window_size, mode='valid')

# Plot iterations per episode
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(episode_iterations) + 1), episode_iterations, marker='o', label='Iterations per Episode')
plt.plot(range(window_size, len(episode_iterations) + 1), moving_average, color='red', label=f'Moving Average (window size={window_size})', linewidth=2)
plt.title('Iterations per Episode with Moving Average')
plt.xlabel('Episode')
plt.ylabel('Iterations (Log Scale)')
plt.legend()
plt.grid(True)
plt.show()
