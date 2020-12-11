import sys, os
if '../' not in sys.path:
    sys.path.append('../')
from source.agent import Agent
from source.data_structures import Dataset
import numpy as np
import pickle
import copy
from tqdm import tqdm

n_episodes_to_load = None

def evaluate_policy(agent, transition_function, reward_function):
    v_old = np.zeros(agent.n_states)
    it = 0
    while True:
        it += 1
        v_new = np.zeros(agent.n_states)
        delta = 0
        for s in range(agent.n_states):
            if s == 15:
                continue
            v_fn = 0
            action_probs = agent.act(s, return_probs=True)
            for a in range(agent.n_actions):
                s_prime = transition_function[s][a]
                # r = np.random.normal(reward_function[s][a][0], reward_function[s][a][1])
                r = reward_function[s][a][0]
                if s_prime is not None:
                    v_fn += action_probs[a] * (r + 0.95 * v_old[s_prime])
                else:
                    v_fn += action_probs[a] * r
            delta = max(delta, np.abs(v_fn - v_old[s]))
            v_new[s] = v_fn
        v_old = v_new
        # if it % 100 == 0:
        #     print(f'Iter: {it}, Delta" {delta}')
        if delta < 0.01 or it > 10000:
            break
    return v_old


if __name__ == '__main__':
    # np.random.seed(RANDOM_SEED)
    if os.path.isfile('transition_function.pkl'):
        with open('transition_function.pkl', 'rb') as f:
            transition_function = pickle.load(f)
        with open('reward_function.pkl', 'rb') as f:
            reward_function = pickle.load(f)
    else:
        f_name = 'dataset'
        if n_episodes_to_load is not None:
            f_name += f'_{n_episodes_to_load}'
        f_name += '.pkl'
        if os.path.isfile(f_name):
            with open(f_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = Dataset()
            dataset.build_dataset(n_episodes_to_load)
            with open(f_name, 'wb') as f:
                pickle.dump(dataset, f)

        transition_function = []
        reward_function = []
        for s in range(18):
            transition_function.append([])
            reward_function.append([])
            for a in range(4):
                transition_function[s].append([])
                reward_function[s].append([])
        for episode in tqdm(dataset.episodes):
            for i in range(len(episode.actions)):
                s = episode.states[i]
                a = episode.actions[i]
                r = episode.rewards[i]
                try:
                    s_prime = episode.states[i + 1]
                except IndexError:
                    s_prime = None
                transition_function[s][a] = s_prime
                reward_function[s][a].append(r)
        for s in range(18):
            for a in range(4):
                reward_function[s][a] = (np.mean(reward_function[s][a]), np.std(reward_function[s][a]))
        with open('transition_function.pkl', 'wb') as f:
            pickle.dump(transition_function, f)
        with open('reward_function.pkl', 'wb') as f:
            pickle.dump(reward_function, f)

    agent = Agent(18, 4, delta=0.05 / 2., sigma=0.1, c=1.45)
    success = []
    for i in range(1, 101):
        print(f'Policy: {i}')
        agent.load_policy(i)
        v_est = evaluate_policy(agent, transition_function, reward_function)
        print(f'Expected Discounted Reward: {v_est[17]}')
        print('v est')
        print(v_est)
        if v_est[17] > 1.41537:
            success.append(1)
        else:
            success.append(0)
        print()
    print(f'Success Rate: {np.mean(success)}')

