import numpy as np
from tqdm import tqdm


class Episode:
    def __init__(self, states, actions, rewards, pb_sas, gamma=0.99):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.pb_sas = pb_sas
        self.gamma = gamma

    def __len__(self):
        return len(self.rewards)


class Dataset:
    def __init__(self, n_episodes_to_load=None):
        if n_episodes_to_load is None:
            n_episodes_to_load = np.inf
        self.n_episodes_to_load = n_episodes_to_load
        self.episodes = []

    def build_dataset(self, n_episodes_to_load):
        episodes = []
        with open('../data.csv', 'r') as f:
            n_episodes = int(f.readline())
            for _ in tqdm(range(min(n_episodes_to_load, n_episodes))):
                episode_length = int(f.readline())
                states, actions, rewards, pb_sas = [], [], [], []
                for i in range(episode_length):
                    step = f.readline()
                    s, a, r, pb_sa = step.split(',')
                    states.append(int(s))
                    actions.append(int(a))
                    rewards.append(float(r))
                    pb_sas.append(float(pb_sa))
                episodes.append(Episode(states, actions, rewards, pb_sas))
        self.episodes = np.array(episodes)

    def __len__(self):
        return len(self.episodes)