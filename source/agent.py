import numpy as np
from source.data_structures import Episode, Dataset
from scipy.stats import t
import cma
from multiprocessing import Pool


def parallel_batch_pdis_h(episode, policies, is_tabular, gamma):
    horizon = len(episode)
    expected_rewards = np.zeros(len(policies))
    cur_gamma = 1
    importance_weights = np.ones(len(policies))
    for i in range(1, horizon + 1):
        s = episode.states[i - 1]
        a = episode.actions[i - 1]
        if is_tabular:
            pe = policies[:, s, :]
        else:
            pe = np.dot(s, policies).squeeze()
        pe = (np.exp(pe) / np.exp(pe).sum(1)[:, None])[:, a]
        pb = episode.pb_sas[i - 1]
        importance_weights *= pe / pb
        expected_rewards += cur_gamma * episode.rewards[i - 1] * importance_weights
        cur_gamma *= gamma
    return expected_rewards


class Agent:
    def __init__(self, n_states, n_actions, is_tabular=True, gamma=0.95, c=1.42, delta=0.05, sigma=1.0, r_max=None,
                 r_min=None, seed=None, accelerate=False):
        self.n_states = n_states
        self.n_actions = n_actions
        self.is_tabular = is_tabular
        self.gamma = gamma
        # self.policy = np.random.normal(0, 1, (n_states, n_actions))  # Final project evaluation specific
        self.policy = np.zeros((n_states, n_actions))  # Final project evaluation specific
        self.c = c  # For the final project evaluation
        self.delta = delta
        self.safety_data = None
        self.candidate_data = None
        self.policy_idx = 1
        self.sigma = sigma
        self.seed = seed
        self.es = cma.CMAEvolutionStrategy(np.zeros(self.policy.flatten().shape), self.sigma)
        self.r_max = r_max
        self.r_min = r_min
        self.accelerate = accelerate

    def act(self, s, return_probs=False, policy=None):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.is_tabular:
            if s >= self.n_states:
                if policy is None:
                    action_probs = self.policy.mean(0)
                else:
                    action_probs = policy.mean(0)
            else:
                if policy is None:
                    action_probs = self.policy[s, :]
                else:
                    action_probs = policy[s, :]
        else:
            if policy is None:
                action_probs = np.dot(s, self.policy).squeeze()
            else:
                action_probs = np.dot(s, policy).squeeze()

        action_probs = np.exp(action_probs) / np.exp(action_probs).sum()
        if not return_probs:
            return np.random.choice(np.arange(self.n_actions), 1, action_probs)
        else:
            return action_probs

    def update(self, safety_data, candidate_data, ngen=1, write=True):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.safety_data = safety_data
        self.candidate_data = candidate_data
        # candidates = self.policy + perturbations
        for _ in range(ngen):
            candidates = self.es.ask()
            candidate_policies = self.policy + np.array(candidates).reshape((-1, self.n_states, self.n_actions))
            pdis, left_side, did_pass = self.safety_test(self.candidate_data, is_candidate_data=True, pi_e=None, policies=candidate_policies)
            # solutions = -1 * pdis + (1 - did_pass.astype(int) * (self.c - left_side) * 1000)
            solutions = -1 * pdis * (left_side - self.c)
            self.es.tell(candidates, solutions)
            # self.es.disp()
        # pi_e = self.es.result.xbest.reshape(self.n_states, self.n_actions)
        pi_e = candidate_policies[np.argmin(solutions)]
        _, test_left_side, test_did_pass = self.safety_test(self.safety_data, pi_e=pi_e, verbose=True)

        if test_did_pass:
            print(f'Policy passed with score {test_left_side} >= {self.c}')
            if write:
                self._write_policy(pi_e)
            self.policy = pi_e
            self.policy_idx += 1
            self.es = cma.CMAEvolutionStrategy(np.zeros(self.policy.flatten().shape), self.sigma)

        else:
            print(f'Policy failed with score {test_left_side} < {self.c}')
        return test_did_pass

    def _write_policy(self, p):
        with open(f'../policy_{self.policy_idx}.txt', 'w') as f:
            for x in p.reshape(-1, 1).squeeze():
                f.write(f'{str(x)}\n')

    def load_policy(self, p):
        a = []
        with open(f'../policy_{p}.txt', 'r') as f:
            lines = f.readlines()
        for l in lines:
            a.append(float(l))
        self.policy = np.array(a).reshape(self.n_states, self.n_actions)

    def safety_test(self, data: Dataset, is_candidate_data=False, pi_e=None, verbose=False, policies=None):
        pdis, sigma = self.pdis_d(data, pi_e=pi_e, policies=policies)
        subtract_term = sigma / np.sqrt(len(self.safety_data))
        if is_candidate_data:
            subtract_term *= 2
        tinv = t.isf(self.delta, len(self.safety_data) - 1)

        left_side = pdis - subtract_term * tinv
        did_pass = left_side >= self.c
        return pdis, left_side, did_pass

    def pdis_h(self, episode: Episode, pi_e=None):
        horizon = len(episode)
        expected_reward = 0
        cur_gamma = 1
        importance_weight = 1
        for i in range(1, horizon+1):
            s = episode.states[i - 1]
            a = episode.actions[i - 1]
            pe = self.act(s, return_probs=True, policy=pi_e)[a]
            pb = episode.pb_sas[i - 1]
            importance_weight *= pe/pb
            expected_reward += cur_gamma * episode.rewards[i - 1] * importance_weight
            cur_gamma *= self.gamma
        if self.r_max is not None and self.r_min is not None:
            expected_reward = (expected_reward - self.r_min) / (self.r_max - self.r_min)
        return expected_reward

    def batch_pdis_h(self, episode, policies):
        horizon = len(episode)
        expected_rewards = np.zeros(len(policies))
        cur_gamma = 1
        importance_weights = np.ones(len(policies))
        for i in range(1, horizon+1):
            s = episode.states[i - 1]
            a = episode.actions[i - 1]
            if self.is_tabular:
                pe = policies[:, s, :]
            else:
                pe = np.dot(s, policies).squeeze()
            pe = (np.exp(pe) / np.exp(pe).sum(1)[:, None])[:, a]
            pb = episode.pb_sas[i - 1]
            importance_weights *= pe/pb
            expected_rewards += cur_gamma * episode.rewards[i - 1] * importance_weights
            cur_gamma *= self.gamma
        if self.r_max is not None and self.r_min is not None:
            expected_rewards = (expected_rewards - self.r_min) / (self.r_max - self.r_min)
        return expected_rewards

    def pdis_d(self, data: Dataset, pi_e=None, policies=None):
        is_estimates = []
        if self.accelerate:
            with Pool() as pool:
                if policies is not None:
                    is_estimates = pool.starmap_async(parallel_batch_pdis_h, [(ep, policies, self.is_tabular, self.gamma) for ep in data.episodes]).get()
                else:
                    is_estimates = pool.starmap_async(parallel_batch_pdis_h, [(ep, np.array([pi_e]), self.is_tabular, self.gamma) for ep in data.episodes]).get()
        else:
            for episode in data.episodes:
                if policies is not None:
                    is_estimates.append(self.batch_pdis_h(episode, policies))
                else:
                    is_estimates.append(self.pdis_h(episode, pi_e))
        if policies is not None:
            mu = np.array(is_estimates).mean(0)
            sigma = np.array(is_estimates).std(0)
        else:
            mu = np.mean(is_estimates)
            sigma = np.sum([(x - mu)**2 for x in is_estimates])
            sigma /= (len(self.safety_data) - 1)
            sigma = np.sqrt(sigma)
        # print(mu)

        return mu, sigma

    def expected_discounted_return(self, data: Dataset):
        gt_estimates = []
        for episode in data.episodes:
            horizon = len(episode)
            expected_reward = 0
            cur_gamma = 1
            for i in range(1, horizon + 1):
                expected_reward += cur_gamma * episode.rewards[i - 1]
                cur_gamma *= self.gamma
            gt_estimates.append(expected_reward)
        return gt_estimates