import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from pbrl.dqn import DQNAgent
from pbrl.replay_buffer import ReplayBuffer


class QNet(torch.nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.fc1 = torch.nn.Linear(observation_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    episodes = 300
    sync_interval = 20
    reward_historys = []
    num_tests = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    for test in tqdm(range(num_tests)):
        q_function = QNet(env.observation_space.shape[0], env.action_space.n).to(device)
        optimizer = torch.optim.Adam(q_function.parameters(), lr=1e-2)
        agent = DQNAgent(
            q_function=q_function,
            optimizer=optimizer,
            replay_buffer=ReplayBuffer(10000, 32),
            gamma=0.99,
            epsilon=0.1,
        )
        reward_history = []
        for episode in tqdm(range(episodes)):
            state, info = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, done, next_state)
                state = next_state
                total_reward += reward
            reward_history.append(total_reward)
            if episode % sync_interval == 0:
                if episode == 0:
                    continue
                agent.sync_target()
        reward_historys.append(reward_history)

    reward_historys = np.array(reward_historys)
    mean_reward_history = reward_historys.mean(axis=0)
    # plt.plot(mean_reward_history)
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.savefig("cartpole_dqn.png")
    # plt.show()
