from collections import deque
import random
import torch


class ReplayBuffer:
    """
    A replay buffer for storing and sampling experiences.

    Args:
        buffer_size (int): the maximum size of the buffer.
        batch_size (int): the size of the batch to sample.
    """

    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the replay buffer.

        Args:
            state (np.ndarray): the current state.
            action (int): the action taken.
            reward (float): the reward obtained.
            next_state (np.ndarray): the resulting state.
            done (bool): whether the episode has ended.
        """
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        """
        Sample a batch of experiences from the replay buffer.

        Returns:
            state (torch.Tensor): the current state.
            action (torch.Tensor): the action taken.
            reward (torch.Tensor): the reward obtained.
            next_state (torch.Tensor): the resulting state.
            done (torch.Tensor): whether the episode has ended.
        """
        data = random.sample(self.buffer, self.batch_size)
        state = torch.stack([d[0] for d in data])
        action = torch.tensor([d[1] for d in data])
        reward = torch.tensor([d[2] for d in data])
        next_state = torch.stack([d[3] for d in data])
        done = torch.tensor([d[4] for d in data]).to(torch.int32)
        return state, action, reward, next_state, done