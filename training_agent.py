import gym
import numpy as np
from maze_env import SimpleMazeEnv
import json
import os

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.5  # Exploration-exploitation trade-off

# Create a simple maze environment
# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        """
        The `replay` method trains the agent by sampling a batch of experiences from memory:
        - It updates the Q-value for each action using the Bellman equation:
        \[
        Q(s, a) = r + \gamma \max_a' Q(s', a')
        \]
        where \( r \) is the reward, \( \gamma \) is the discount factor, and \( s' \) is the next state.
        - The neural network is optimized using backpropagation to minimize the loss between the target Q-value and the predicted Q-value.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state).detach())
            target_f = self.model(state)
            target_f[0][0][action] = target
            loss = self.criterion(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create maze environment
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0]
]
env = SimpleMazeEnv(maze)

# Q-table initialization
q_table = np.zeros((env.observation_space.n, env.action_space.n))

def state_to_index(state):
    # Convert (x, y) state tuple to a single integer index
    x, y = state
    index = x * len(maze[0]) + y
    return index

# Training parameters
num_episodes = 100
episode_iterations = []  # To store number of iterations per episode
q_table_history = []  # To store Q-table after each episode

# Training the agent
for episode in range(num_episodes):
    state = env.reset()
    state_index = state_to_index(state)
    done = False
    iterations = 0

    while not done:
        # Choose action using epsilon-greedy policy
        randy = np.random.random()
        if randy < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            action = np.argmax(q_table[state_index])  # Exploitation

        next_state, reward, done, _ = env.step(action)
        next_state_index = state_to_index(next_state)

        # Q-table update using Bellman equation
        q_table[state_index][action] += alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index][action])
        state_index = next_state_index

        iterations += 1

    episode_iterations.append(iterations)
    q_table_history.append(q_table.copy())  # Save a copy of Q-table after each episode
    print(f"Episode {episode + 1} completed. Iterations: {iterations}")

# Save Q-table history and episode iteration data to a single JSON file
output_data = {
    'q_table_history': [q.tolist() for q in q_table_history],  # Convert each Q-table to list for JSON serialization
    'episode_iterations': episode_iterations
}

output_folder = 'output'  # Output folder for saving data
os.makedirs(output_folder, exist_ok=True)
file_path = os.path.join(output_folder, 'q_learning_data.json')

with open(file_path, 'w') as f:
    json.dump(output_data, f)

print(f"Q-table history and episode iteration data saved to {file_path}")
print("Training complete.")
