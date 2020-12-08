import sys, os
from source.agent import Agent
from source.data_structures import Dataset, Episode
import numpy as np
import pickle
import copy
import gym
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

# RANDOM_SEED = 42
env = gym.make('MountainCar-v0')
poly = PolynomialFeatures(3)


def generate_dataset(env, agent, n_rollouts):
    dataset = Dataset()
    for _ in tqdm(range(n_rollouts)):
        s = env.reset()
        s = poly.fit_transform(s.reshape(1, -1))
        done = False
        states, actions, rewards, pb_sas = [], [], [], []
        while not done:
            action_probs = agent.act(s, return_probs=True)
            a = np.random.choice(np.arange(agent.n_actions), 1, p=action_probs)[0]
            next_state, r, done, _ = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            pb_sas.append(action_probs[a])
            s = poly.fit_transform(next_state.reshape(1, -1))

        rewards[-1] = rewards[-1] * 10
        dataset.episodes.append(Episode(states, actions, rewards, pb_sas))
    return dataset


if __name__ == '__main__':
    # np.random.seed(RANDOM_SEED)
    s = env.reset()
    n_states = poly.fit_transform(s.reshape(1, -1))
    agent = Agent(n_states.shape[1], env.action_space.n, delta=0.2, sigma=0.01, is_tabular=False)
    agent.c = -np.inf
    mean_return = 0
    did_improve = []
    safety_dataset = generate_dataset(env, agent, 1000)
    candidate_dataset = generate_dataset(env, agent, 1000)
    ngen = 1
    for epoch in range(1000):
        print(f'Epoch: {epoch}')
        print('---------------')

        did_pass = agent.update(safety_dataset, candidate_dataset, 1, write=False)

        if did_pass:
            eval_dataset = generate_dataset(env, agent, 500)
            gt_estimates = agent.expected_discounted_return(eval_dataset)
            next_mean_return = np.mean(gt_estimates)
            print(f'Average discounted reward: {next_mean_return}')
            did_improve.append(next_mean_return > agent.c)
            agent.c = next_mean_return * 1.0
            mean_return = next_mean_return

            safety_dataset = generate_dataset(env, agent, 1000)
            candidate_dataset = generate_dataset(env, agent, 1000)
        else:
            ngen += 1
        # if ngen > 50:
        #     agent.reset_es()
        #     ngen = 1

        print(f'Current success rate: {np.mean(np.array(did_improve).astype(int))}')
        print(f'Current policy iteration: {agent.policy_idx}')
        print()
