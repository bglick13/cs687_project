from source.agent import Agent
from source.data_structures import Dataset, Episode
import numpy as np

import gym

# RANDOM_SEED = 42
env = gym.make('FrozenLake-v0')


def generate_dataset(env, agent, n_rollouts):
    dataset = Dataset()
    success = []
    for _ in range(n_rollouts):
        s = env.reset()
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
            s = next_state
        if rewards[-1] == 0:
            # rewards[-1] = -1
            success.append(0)

        else:
            success.append(1)
        rewards[-1] = rewards[-1] * 10
        dataset.episodes.append(Episode(states, actions, rewards, pb_sas))
    print(f'Success rate: {np.mean(success)}')
    return dataset


if __name__ == '__main__':
    # np.random.seed(RANDOM_SEED)

    agent = Agent(env.observation_space.n, env.action_space.n, delta=0.25, sigma=0.01)
    agent.c = -10
    mean_return = 0
    did_improve = []
    safety_dataset = generate_dataset(env, agent, 5000)
    candidate_dataset = generate_dataset(env, agent, 5000)
    ngen = 1
    for epoch in range(1000):
        print(f'Epoch: {epoch}')
        print('---------------')

        did_pass = agent.update(safety_dataset, candidate_dataset, 1, write=False)

        if did_pass:
            eval_dataset = generate_dataset(env, agent, 5000)
            gt_estimates = agent.expected_discounted_return(eval_dataset)
            next_mean_return = np.mean(gt_estimates)
            print(f'Average discounted reward: {next_mean_return}')
            did_improve.append(next_mean_return > agent.c)
            agent.c = next_mean_return * 1.0
            mean_return = next_mean_return

            safety_dataset = generate_dataset(env, agent, 5000)
            candidate_dataset = generate_dataset(env, agent, 5000)
        else:
            ngen += 1

        print(f'Current success rate: {np.mean(np.array(did_improve).astype(int))}')
        print(f'Current policy iteration: {agent.policy_idx}')
        print()
