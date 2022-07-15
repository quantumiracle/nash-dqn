import numpy as np
import random, copy, math
import torch
import torch.nn.functional as F
import torch.optim as optim
from agent import Agent
from common.storage import ReplayBuffer
from common.networks import NetBase, get_model

class EpsilonScheduler():
    def __init__(self, eps_start, eps_final, eps_decay):
        """A scheduler for epsilon-greedy strategy.

        :param eps_start: starting value of epsilon, default 1. as purely random policy 
        :type eps_start: float
        :param eps_final: final value of epsilon
        :type eps_final: float
        :param eps_decay: number of timesteps from eps_start to eps_final
        :type eps_decay: int
        """
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.epsilon = self.eps_start
        self.ini_frame_idx = 0
        self.current_frame_idx = 0
        self.restart_period = 300*10000 # periodically reset the scheduler

    def reset(self, ):
        """ Reset the scheduler """
        self.ini_frame_idx = self.current_frame_idx

    def step(self, frame_idx):
        """
        The choice of eps_decay:
        ------------------------
        start = 1
        final = 0.01
        decay = 10**6  # the decay steps can be 1/10 over all steps 10000*1000
        final + (start-final)*np.exp(-1*(10**7)/decay)
        
        => 0.01

        """
        self.current_frame_idx = frame_idx
        delta_frame_idx = self.current_frame_idx - self.ini_frame_idx
        self.epsilon = self.eps_final + (self.eps_start - self.eps_final) * math.exp(-1. * delta_frame_idx / self.eps_decay)
    
        # if delta_frame_idx > self.restart_period:
        #     self.reset()

    def get_epsilon(self):
        return self.epsilon


class DQN(Agent):
    """
    DQN algorithm
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self._init_model(env, args)
        
        if args.num_process > 1:
            self.model.share_memory()
            self.target.share_memory()
            self.buffer = args.add_components['replay_buffer']
        else:
            self.buffer = ReplayBuffer(int(float(args.algorithm_spec['replay_buffer_size'])), \
                args.algorithm_spec['multi_step'], args.algorithm_spec['gamma'], args.num_envs, args.batch_size) # first float then int to handle the scientific number like 1e5

        self.update_target(self.model, self.target)

        self.optimizer = optim.Adam(self.model.parameters(), lr=float(args.learning_rate))
        self.epsilon_scheduler = EpsilonScheduler(args.algorithm_spec['eps_start'], args.algorithm_spec['eps_final'], args.algorithm_spec['eps_decay'])
        self.schedulers.append(self.epsilon_scheduler)

        self.gamma = float(args.algorithm_spec['gamma'])
        self.multi_step = args.algorithm_spec['multi_step'] 
        self.target_update_interval = args.algorithm_spec['target_update_interval']

        self.update_cnt = 1

    def _init_model(self, env, args):
        self.model = self._select_type(env, args).to(self.device)
        self.target = copy.deepcopy(self.model).to(self.device)

    def _select_type(self, env, args):
        if args.num_envs == 1:
            if args.algorithm_spec['dueling']:
                model = DuelingDQN(env, args.net_architecture)
            else:
                model = DQNBase(env, args.net_architecture)
        else:
            if args.algorithm_spec['dueling']:
                model = ParallelDuelingDQN(env, args.net_architecture, args.num_envs)
            else:
                model = ParallelDQN(env, args.net_architecture, args.num_envs)
        return model

    def reinit(self, nets_init=False, buffer_init=True, schedulers_init=True):
        if nets_init:
            self.model.reinit()  # reinit the networks seem to hurt the overall learning performance
            self.target.reinit()
            self.update_target(self.model, self.target)
        if buffer_init:
            self.buffer.clear()
        if schedulers_init:
            for scheduler in self.schedulers:
                scheduler.reset()

    def choose_action(
        self, 
        state, 
        Greedy = False, 
        epsilon = None
        ):
        """Choose action give state.

        :param state: observed state from the agent
        :type state: List[StateType]
        :param Greedy: whether adopt greedy policy (no randomness for exploration) or not, defaults to False
        :type Greedy: bool, optional
        :param epsilon: parameter value for \epsilon-greedy, defaults to None
        :type epsilon: Union[float, None], optional
        :return: the actions
        :rtype: List[ActionType]
        """
        if Greedy:
            epsilon = 0.
        elif epsilon is None:
            epsilon = self.epsilon_scheduler.get_epsilon()
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state).to(self.device)
        action = self.model.choose_action(state, epsilon)

        return action

    def store(self, sample) -> None:
        """ Store samples in buffer.

        :param sample: a list of samples from different environments (if using parallel env)
        :type sample: SampleType
        """ 
        self.buffer.push(sample)

    @property
    def ready_to_update(self):
        return True if self.buffer.get_len() > self.batch_size else False

    def update(self):
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(np.float32(done)).to(self.device)

        # Q-Learning with target network
        q_values = self.model(state)
        target_next_q_values = self.target(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = target_next_q_values.max(1)[0]

        expected_q_value = reward + (self.gamma ** self.multi_step) * next_q_value * (1 - done)
        # additional value normalization (this effectively prevent increasing Q/loss value)
        expected_q_value =  (expected_q_value - expected_q_value.mean(dim=0)) / (expected_q_value.std(dim=0) + 1e-6)

        # Huber Loss
        loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')  # slimevolley env only works with this!
        # loss = F.mse_loss(q_value, expected_q_value.detach())

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_cnt % self.target_update_interval == 0:
            self.update_target(self.model, self.target)
        self.update_cnt += 1

        return loss.detach().item()

    def save_model(self, path):
        try:  # for PyTorch >= 1.7 to be compatible with loading models from any lower version
            torch.save(self.model.state_dict(), path+'_model', _use_new_zipfile_serialization=False) 
            torch.save(self.target.state_dict(), path+'_target', _use_new_zipfile_serialization=False)
        except:  # for lower versions
            torch.save(self.model.state_dict(), path+'_model')
            torch.save(self.target.state_dict(), path+'_target')

    def load_model(self, path, eval=True):
        self.model.load_state_dict(torch.load(path+'_model', map_location=torch.device(self.device)))
        self.target.load_state_dict(torch.load(path+'_target', map_location=torch.device(self.device)))

        if eval:
            self.model.eval()
            self.target.eval()

class DQNBase(NetBase):
    """Basic Q network

    :param env: env object
    :type env: object
    :param net_args: network architecture arguments
    :type net_args: dict
    """
    def __init__(self, env, net_args):
        super().__init__(env.observation_space, env.action_space)
        self._construct_net(env, net_args)

    def _construct_net(self, env, net_args):
        if len(self._observation_shape) <= 1: # not image
            self.net = get_model('mlp')(env.observation_space, env.action_space, net_args, model_for='discrete_q')
        else:
            self.net = get_model('cnn')(env.observation_space, env.action_space, net_args, model_for='discrete_q')
    
    def reinit(self, ):
        self.net.reinit()

    def forward(self, x):
        return self.net(x)

    def choose_action(self, state, epsilon=0.):
        """Choose action acoording to state.

        :param state: state/observation input
        :type state:  torch.Tensor
        :param epsilon: epsilon for epsilon-greedy, defaults to 0.
        :type epsilon: float, optional
        :return: action
        :rtype: int or np.ndarray
        """        
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_value = self.net(state)
                action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self._action_shape)
        return action


class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env, net_args, **kwargs):
        super().__init__(env, net_args, **kwargs)
        self._construct_net(env, net_args)
    
    def _construct_net(self, env, net_args):
        # Here I use separate networks for advantage and value heads 
        # due to the usage of internal network builder, they should use
        # a shared network body with two heads.
        if len(self._observation_shape) <= 1: # not image
            self.advantage = get_model('mlp')(env.observation_space, env.action_space, net_args, model_for='discrete_q')
            self.value = get_model('mlp')(env.observation_space, env.action_space, net_args, model_for='discrete_q')
        else:  
            self.advantage = get_model('cnn')(env.observation_space, env.action_space, net_args, model_for='discrete_q')
            self.value = get_model('cnn')(env.observation_space, env.action_space, net_args, model_for='discrete_q')

    def reinit(self, ):
        self.advantage.reinit()
        self.value.reinit()

    def net(self, x):
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)

class ParallelDQN(DQNBase):
    """ DQN for parallel env sampling

    :param env: env object
    :type env: object
    :param net_args: network architecture arguments
    :type net_args: dict
    :param number_envs: number of environments
    :type number_envs: int
    :param kwargs: arbitrary keyword arguments.
    :type kwargs: dict
    """
    def __init__(self, env, net_args, number_envs, **kwargs):
        super(ParallelDQN, self).__init__(env, net_args, **kwargs)
        self.number_envs = number_envs

    def choose_action(self, state, epsilon):
        """Choose action acoording to state.

        :param state: state/observation input
        :type state:  torch.Tensor
        :param epsilon: epsilon for epsilon-greedy, defaults to 0.
        :type epsilon: float, optional
        :return: action
        :rtype: int or np.ndarray
        """  
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                q_value = self.net(state)
                action = q_value.max(1)[1].detach().cpu().numpy()
        else:
            action = np.random.randint(self._action_shape, size=self.number_envs)
        return action

class ParallelDuelingDQN(DuelingDQN, ParallelDQN):
    """ DuelingDQN for parallel env sampling

    :param env: env object
    :type env: object
    :param net_args: network architecture arguments
    :type net_args: dict
    :param number_envs: number of environments
    :type number_envs: int
    :param kwargs: other arguments
    :type kwargs: dict

    Note: for mulitple inheritance, see a minimal example:

    .. code-block:: python

        class D:
            def __init__(self,):
                super(D, self).__init__()
                self.a=1
            def f(self):
                pass          
            def f1(self):
                pass
        class A(D):
            def __init__(self,):
                super(A, self).__init__()
                self.a=1     
            def f1(self):
                self.a+=2
                print(self.a)          
        class B(D):
            def __init__(self,):
                super(B, self).__init__()
                self.a=1
            def f(self):
                self.a-=1
                print(self.a)          
        class C(B,A):  # the order indicates piority for overwritting
            def __init__(self,):
                super(C, self).__init__()   
        c=C()
        c.f1() 
        => 3
    
    """
    def __init__(self, env, net_args, number_envs):
        super(ParallelDuelingDQN, self).__init__(env=env, net_args=net_args, number_envs=number_envs)

