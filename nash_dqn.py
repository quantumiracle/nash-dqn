import numpy as np
import gym
import operator
import random, copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from common.networks import get_model
from dqn import DQN, DQNBase
from equilibrium_solver import NashEquilibriumECOSSolver

class NashDQN(DQN):
    """
    Nash-DQN algorithm
    """
    def __init__(self, env, args):
        super().__init__(env, args)

        if args.num_process > 1:
            self.model.share_memory()
            self.target.share_memory()
        self.num_agents = env.num_agents[0] if isinstance(env.num_agents, list) else env.num_agents
        self.env = env
        self.args = args

        # don't forget to instantiate an optimizer although there is one in DQN
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(args.learning_rate))

    def _init_model(self, env, args):
        """Overwrite DQN's models

        :param env: environment
        :type env: object
        :param args: arguments
        :type args: dict
        """
        self.model = NashDQNBase(env, args.net_architecture, args.num_envs, two_side_obs = args.marl_spec['global_state']).to(self.device)
        print(self.model)
        self.target = copy.deepcopy(self.model).to(self.device)

    def choose_action(self, state, Greedy=False, epsilon=None):
        if Greedy:
            epsilon = 0.
        elif epsilon is None:
            epsilon = self.epsilon_scheduler.get_epsilon()
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state).to(self.device)
        if self.args.ram:
            if self.args.num_envs == 1: # state: (agents, state_dim)
                state = state.unsqueeze(0).view(1, -1) # change state from (agents, state_dim) to (1, agents*state_dim)
            else: # state: (agents, envs, state_dim)
                state = torch.transpose(state, 0, 1) # to state: (envs, agents, state_dim)
                state = state.view(state.shape[0], -1) # to state: (envs, agents*state_dim)
        else:  # image-based input
            if self.args.num_envs == 1: # state: (agents, C, H, W)
                state = state.unsqueeze(0).view(1, -1, state.shape[-2], state.shape[-1])  #   (1, agents*C, H, W)

            else: # state: (agents, envs, C, H, W)
                state = torch.transpose(state, 0, 1) # state: (envs, agents, C, H, W)
                state = state.view(state.shape[0], -1, state.shape[-2], state.shape[-1]) # state: (envs, agents*C, H, W)

        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                q_values = self.model(state).detach().cpu().numpy()  # needs state: (batch, agents*state_dim)

            try: # nash computation may report error and terminate the process
                actions, dists, ne_vs = self.compute_nash(q_values)
            except:
                print("Invalid nash computation.")
                actions = np.random.randint(self.action_dim, size=(state.shape[0], self.num_agents))

        else:
            actions = np.random.randint(self.action_dim, size=(state.shape[0], self.num_agents))  # (envs, agents)
        
        if self.args.num_envs == 1:
            actions = actions[0]  # list of actions to its item
        else:
            actions = np.array(actions).T  # to shape: (agents, envs, action_dim)
        return actions

    def compute_nash(self, q_values, update=False):
        q_tables = q_values.reshape(-1, self.action_dim,  self.action_dim)
        all_actions = []
        all_dists = []
        all_ne_values = []

        for q_table in q_tables:
            dist, value = NashEquilibriumECOSSolver(q_table)
            all_dists.append(dist)
            all_ne_values.append(value)

        if update:
            return all_dists, all_ne_values
        else:
            # Sample actions from Nash strategies
            for ne in all_dists:
                actions = []
                for dist in ne:  # iterate over agents
                    try:
                        sample_hist = np.random.multinomial(1, dist)  # return one-hot vectors as sample from multinomial
                    except:
                        print('Not a valid distribution from Nash equilibrium solution: ', dist)
                    a = np.where(sample_hist>0)
                    actions.append(a)
                all_actions.append(np.array(actions).reshape(-1))

            return np.array(all_actions), all_dists, all_ne_values

    def update(self):
        DoubleTrick = False
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.IntTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(np.float32(done)).to(self.device)

        # Q-Learning with target network
        q_values = self.model(state)
        target_next_q_values_ = self.model(next_state) if DoubleTrick else self.target(next_state)
        target_next_q_values = target_next_q_values_.detach().cpu().numpy()

        action_ = torch.LongTensor([a[0]*self.action_dim+a[1] for a in action]).to(self.device)
        q_value = q_values.gather(1, action_.unsqueeze(1)).squeeze(1)

        # solve matrix Nash equilibrium
        try: # nash computation may encounter error and terminate the process
            next_dist, next_q_value = self.compute_nash(target_next_q_values, update=True)
        except: 
            print("Invalid nash computation.")
            next_q_value = np.zeros_like(reward)

        if DoubleTrick: # calculate next_q_value using double DQN trick
            next_dist = np.array(next_dist)  # shape: (#batch, #agent, #action)
            target_next_q_values = target_next_q_values.reshape((-1, self.action_dim, self.action_dim))
            left_multi = np.einsum('na,nab->nb', next_dist[:, 0], target_next_q_values) # shape: (#batch, #action)
            next_q_value = np.einsum('nb,nb->n', left_multi, next_dist[:, 1]) 

        next_q_value  = torch.FloatTensor(next_q_value).to(self.device)

        expected_q_value = reward + (self.gamma ** self.multi_step) * next_q_value * (1 - done)

        # Huber Loss
        loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none')
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_cnt % self.target_update_interval == 0:
            self.update_target(self.model, self.target)
        self.update_cnt += 1
        return loss.item()

class NashDQNBase(DQNBase):
    """
    Nash-DQN for parallel env sampling

    parameters
    ---------
    env         environment(openai gym)
    """
    def __init__(self, env, net_args, number_envs=2, two_side_obs=True):
        super().__init__(env, net_args)
        self.number_envs = number_envs
        try:
            if two_side_obs:
                self._observation_shape = tuple(map(operator.add, env.observation_space.shape, env.observation_space.shape)) # double the shape
            else:
                self._observation_shape = env.observation_space.shape
            self._action_shape = (env.action_space.n)**2
        except:
            if two_side_obs:
                self._observation_shape = tuple(map(operator.add, env.observation_space[0].shape, env.observation_space[0].shape)) # double the shape
            else:
                self._observation_shape = env.observation_space[0].shape
            self._action_shape = (env.action_space[0].n)**2
        self._construct_net(env, net_args)

    def _construct_net(self, env, net_args):
            input_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = self._observation_shape)
            output_space = gym.spaces.Discrete(self._action_shape)
            if len(self._observation_shape) <= 1: # not 3d image
                self.net = get_model('mlp')(input_space, output_space, net_args, model_for='discrete_q')
            else:
                self.net = get_model('cnn')(input_space, output_space, net_args, model_for='discrete_q')
