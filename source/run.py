import sys, os
if '../' not in sys.path:
    sys.path.append('../')
from source.agent import Agent
from source.data_structures import Dataset
import numpy as np
import pickle
import copy

n_episodes_to_load = None
if __name__ == '__main__':
    # np.random.seed(RANDOM_SEED)
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

    test_split = 0.5

    batch_size = len(dataset) // 100
    print(f'Batch size: {batch_size}')
    n_train_samples = int(batch_size * (1 - test_split))
    n_test_samples = batch_size - n_train_samples

    for it in range(100):
        train_idxs = np.random.choice(np.arange(batch_size), n_train_samples, replace=False)
        test_idxs = np.array(list(set(np.arange(batch_size)) - set(train_idxs)))

        safety_data = Dataset()
        candidate_data = Dataset()
        batch_episodes = copy.deepcopy(dataset.episodes[it * batch_size: (it + 1) * batch_size])
        candidate_data.episodes = batch_episodes[train_idxs]
        safety_data.episodes = batch_episodes[test_idxs]

        agent = Agent(18, 4, delta=0.05/2., sigma=0.1, c=1.45)
        agent.policy_idx = it + 1
        while True:
            did_pass = agent.update(safety_data, candidate_data, 1)
            if did_pass:
                break
