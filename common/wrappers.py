from supersuit.utils.base_aec_wrapper import PettingzooWrap
from supersuit.utils.wrapper_chooser import WrapperChooser
import gym
from supersuit.utils.make_defaultdict import make_defaultdict
import numpy as np

class aec_reward_lambda(PettingzooWrap):
    def __init__(self, env, change_reward_fn):
        assert callable(
            change_reward_fn
        ), "change_reward_fn needs to be a function. It is {}".format(change_reward_fn)
        self._change_reward_fn = change_reward_fn

        super().__init__(env)

    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        pass

    def reset(self, seed=None, return_info=False, options=None):
        # super().reset(seed=seed, options=options)
        super().reset(seed=seed)
        changed_rewards = self._change_reward_fn(list(self.rewards.values()))
        self.rewards = {
            agent: reward
            for agent, reward in zip(list(self.rewards.keys()), changed_rewards)
        }
        self.__cumulative_rewards = make_defaultdict({a: 0 for a in self.agents})
        self._accumulate_rewards()

    def step(self, action):
        agent = self.env.agent_selection
        super().step(action)
        changed_rewards = self._change_reward_fn(list(self.rewards.values()))
        self.rewards = {
            agent: reward
            for agent, reward in zip(list(self.rewards.keys()), changed_rewards)
        }
        self.__cumulative_rewards[agent] = 0
        self._cumulative_rewards = self.__cumulative_rewards
        self._accumulate_rewards()


class gym_reward_lambda(gym.Wrapper):
    def __init__(self, env, change_reward_fn):
        assert callable(
            change_reward_fn
        ), "change_reward_fn needs to be a function. It is {}".format(change_reward_fn)
        self._change_reward_fn = change_reward_fn

        super().__init__(env)

    def step(self, action):
        obs, rew, done, info = super().step(action)
        return obs, self._change_reward_fn(rew), done, info

def zero_sum_reward_filer(r):
    ## zero-sum filter: 
    # added for making non-zero sum game to be zero-sum, e.g. tennis_v2, pong_v3
    r = np.array(r)
    if np.sum(r) != 0:
        nonzero_idx = np.nonzero(r)[0][0]
        r[1-nonzero_idx] = -r[nonzero_idx]
    return r 

reward_lambda_v1 = WrapperChooser(
    aec_wrapper=aec_reward_lambda, gym_wrapper=gym_reward_lambda
)


class SSVecWrapper():
    """ Wrap after supersuit vector env """
    def __init__(self, env):
        super(SSVecWrapper, self).__init__()
        self.env = env
        if len(env.observation_space.shape) > 1: # image, obs space: (H, W, C) -> (C, H, W)
            old_shape = env.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)
            self.obs_type = 'rgb_image'
        else:
            self.observation_space = env.observation_space
            self.obs_type = 'ram'

        self.action_space = env.action_space
        self.num_agents = env.num_agents
        self.agents = env.agents
        self.true_num_envs = self.env.num_envs//self.env.num_agents
        self.num_envs = self.true_num_envs
    
    @property
    def spec(self):
        return self.env.spec

    def reset(self):
        obs = self.env.reset() 
        if len(self.observation_space.shape) >= 3:
            obs = np.moveaxis(obs, -1, 1) # (N, H, W, C) -> (N, C, H, W)
            obs = obs.reshape(self.true_num_envs, self.env.num_agents, obs.shape[-3], obs.shape[-2], obs.shape[-1])
        else:
            obs = obs.reshape(self.true_num_envs, self.env.num_agents, -1)
        return obs

    def seed(self, seed):
        self.env.seed(seed)

    def render(self, mode='rgb_array'):
        self.env.render(mode)

    def step(self, actions):
        actions = actions.reshape(-1)
        obs, reward, done, info = self.env.step(actions)
        if len(self.observation_space.shape) >= 3:
            obs = np.moveaxis(obs, -1, 1) # (N, H, W, C) -> (N, C, H, W)
            obs = obs.reshape(self.true_num_envs, self.env.num_agents, obs.shape[-3], obs.shape[-2], obs.shape[-1])
        else:
            obs = obs.reshape(self.true_num_envs, self.env.num_agents, -1)
        reward = reward.reshape(self.true_num_envs, self.env.num_agents)
        done = done.reshape(self.true_num_envs, self.env.num_agents)
        info = [info[:self.true_num_envs], info[self.true_num_envs:]]
        return obs, reward, done, info

    def close(self):
        self.env.close()