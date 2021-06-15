import torch
import torch.optim as optim
import torch.nn.functional as F

import time, os
import random
import numpy as np
from tensorboardX import SummaryWriter

from common.utils import epsilon_scheduler, update_target, print_log, load_model, save_model
from common.model import DQN, Policy, ParallelNashDQN
from common.storage import ParallelReplayBuffer, ReservoirBuffer
from common.utils import create_log_dir, print_args, set_global_seeds
from common.wrappers import wrap_pytorch, make_env
from common.arguments import get_args
from common.env import DummyVectorEnv, SubprocVectorEnv
from equilibrium_solver import * 

class ParallelNashAgent():
    def __init__(self, env, args):
        super(ParallelNashAgent, self).__init__()
        self.env = env
        self.num_player = len(env.agents[0])  # env.agents: (envs, agents) when using parallel envs
        self.args = args
        try:
            self.action_dims = self.env.action_space[0].n
        except:
            self.action_dims = self.env.action_space.n
        self.current_model = DQN(env, args, Nash=True).to(args.device)
        self.target_model = DQN(env, args, Nash=True).to(args.device)
        update_target(self.current_model, self.target_model)

        if args.load_model and os.path.isfile(args.load_model):
            self.load_model(model_path)

        self.epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)
        self.replay_buffer = ParallelReplayBuffer(args.buffer_size)
        self.rl_optimizer = optim.Adam(self.current_model.parameters(), lr=args.lr)

    def compute_nash(self, q_values, return_dist=False):
        """
        Return actions as Nash equilibrium of given payoff matrix, shape: [env, agent]
        """
        q_table = q_values.reshape(-1, self.action_dims,  self.action_dims)
        all_actions = []
        all_dists = []
        for qs in q_table:  # iterate over envs
            try:
                ### Select the Nash Equilibrium solver ###
                # ne = NashEquilibriaSolver(qs)
                # ne = ne[0]  # take the first Nash equilibria found
                # print(np.linalg.det(qs))
                # ne = NashEquilibriumSolver(qs)
                # ne = NashEquilibriumLPSolver(qs)
                # ne = NashEquilibriumCVXPYSolver(qs)
                # ne = NashEquilibriumGUROBISolver(qs)
                ne = NashEquilibriumECOSSolver(qs)

            except:  # some cases NE cannot be solved
                print('No Nash solution for: ', np.linalg.det(qs), qs)
                ne = self.num_player*[1./qs.shape[0]*np.ones(qs.shape[0])]  # use uniform distribution if no NE is found
            
            actions = []
                
            all_dists.append(ne)
            for dist in ne:  # iterate over agents
                try:
                    sample_hist = np.random.multinomial(1, dist)  # return one-hot vectors as sample from multinomial
                except:
                    print('Not a valid distribution from Nash equilibrium solution.')
                    print(sum(ne[0]), sum(ne[1]))
                    print(qs, ne)
                    print(dist)
                a = np.where(sample_hist>0)
                actions.append(a)
            all_actions.append(np.array(actions).reshape(-1))
        if return_dist:
            return all_dists
        else:
            return np.array(all_actions)

    # def compute_cce(self, q_values, return_dist=False):
    #     """
    #     Return actions as coarse correlated equilibrium of given payoff matrix, shape: [env, agent]
    #     """
    #     q_table = q_values.reshape(-1, self.action_dims,  self.action_dims)
    #     all_actions = []
    #     all_dists = []
    #     for qs in q_table:  # iterate over envs
    #         try:
    #             _, _, jnt_probs = CoarseCorrelatedEquilibriumLPSolver(qs)

    #         except:  # some cases NE cannot be solved
    #             print('No CCE solution for: ', np.linalg.det(qs), qs)
    #             jnt_probs = 1./(qs.shape[0]*qs.shape[1])*np.ones(qs.shape[0]*qs.shape[1])  # use uniform distribution if no NE is found
            
    #         try:
    #             sample_hist = np.random.multinomial(1, jnt_probs)  # a joint probability matrix for all players
    #         except:
    #             print('Not a valid distribution from Nash equilibrium solution.')
    #             print(sum(jnt_probs), sum(abs(jnt_probs)))
    #             print(qs, jnt_probs)
    #         sample_hist = sample_hist.reshape(self.action_dims,  self.action_dims)
    #         a = np.where(sample_hist>0)  # the actions for two players
    #         all_actions.append(np.array(a).reshape(-1))
    #         all_dists.append(jnt_probs)
    #     if return_dist:
    #         return all_dists
    #     else:
    #         return np.array(all_actions)

    def act(self, states, epsilon):
        states = torch.FloatTensor(states).to(self.args.device)

        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                q_values = self.current_model(states).detach().cpu().numpy()
            if self.args.cce:
                actions = self.compute_cce(q_values)
            else:
                actions = self.compute_nash(q_values) 

        else:
            actions = np.random.randint(self.action_dims, size=(states.shape[0], self.num_player))
        return actions

    # def act_greedy(self, states, epsilon):
    #     """
    #     Take greedy actions with probability 1-epsilon, the 'greedy' means (for zero-sum game):
    #     for player 1, take the action to maximize the Q value in whole table;
    #     for player 2, take the action to minimize the Q value in whole table. 
    #     """ 
    #     states = torch.FloatTensor(states).to(self.args.device)

    #     if random.random() > epsilon:  # NoisyNet does not use e-greedy
    #         with torch.no_grad():
    #             q_values = self.current_model(states).detach().cpu().numpy()
    #             max_q_idx = np.argmax(q_values, axis=-1)   # first dimension of q_values is env
    #             min_q_idx = np.argmin(q_values, axis=-1)
    #             actions = np.stack((max_q_idx // self.action_dims, min_q_idx % self.action_dims), axis=1) # player 1 is row, player 2 is column
    #             # with np.printoptions(threshold=np.inf):
    #             #     print(q_values_, actions)
    #     else:
    #         actions = np.random.randint(self.action_dims, size=(states.shape[0], self.num_player))
    #     return actions

    def save_model(self, model_path):
        torch.save(self.current_model.state_dict(), model_path+f'dqn')
        # torch.save(self.target_model.state_dict(), model_path+f'dqn_target')

    def load_model(self, model_path, eval=False, map_location=None):
        self.current_model.load_state_dict(torch.load(model_path+f'dqn', map_location=map_location))
        # self.target_model.load_state_dict(torch.load(model_path+f'dqn_target', map_location=map_location))
        if eval:
            self.current_model.eval()
            # self.target_model.eval()

def train(env, args, writer, model_path, num_agents=2):
    agent = ParallelNashAgent(env, args)

    # Logging
    length_list = []
    reward_list = [[] for _ in range(num_agents)]
    rl_loss_list = []  # b.c. using the nash Q value, share for two players
    q_list = []
    episode_reward = [0 for _ in range(num_agents)]
    episode_reward_separate = [[0 for _ in range(num_agents)] for _ in range(args.num_envs)] # separate rewards for each env
    tag_interval_length = 0
    prev_time = time.time()
    prev_frame = 1

    # Main Loop
    states =  env.reset()
    t0=time.time()
    [n0, n1] = env.agents[0] # agents name in one env, 2-player game
    for frame_idx in range(1, args.max_frames + 1): # each step contains args.num_envs steps actually due to parallel envs
        t1=time.time()
        epsilon = agent.epsilon_by_frame(frame_idx)
        actions_ = agent.act(states.reshape(states.shape[0], -1), epsilon)  # concate states of all agents
        # actions_ = agent.act_greedy(states.reshape(states.shape[0], -1), epsilon)  # concate states of all agents
        # for qs in q_values:
        #     maxid = np.argmax(qs.reshape(6,6))
        #     actions_.append([maxid//6, maxid%6])
        # actions_ = np.array(actions_)  
        t2=time.time()
        assert num_agents == 2
        actions = [{n0: a0, n1: a1} for a0, a1 in zip(*actions_.T)] 
        next_states, rewards, dones, infos = env.step(actions)
        done = [np.float32(d) for d in dones]

        # states (env, agent, state_dim) -> (env, agent*state_dim), similar for actions_, rewards take the positive one in two agents 
        samples = [[states[j].reshape(-1), actions_[j].reshape(-1), rewards[j, 0], next_states[j].reshape(-1), d] for j, d in enumerate(done) if not np.all(d)]
        agent.replay_buffer.push(samples) 
        
        info = [list(i.values())[1] for i in infos]  # infos is a list of dicts (env) of dicts (agents)
        states = next_states
        # Logging
        for i in range(num_agents):
            episode_reward[i] += np.mean(rewards[:, i])  # mean over envs
            for j in range(args.num_envs):
                episode_reward_separate[j][i] += rewards[j][i]
        tag_interval_length += 1
        # Episode done. Reset environment and clear logging records
        if np.any(done) or tag_interval_length >= args.max_tag_interval:  # TODO if use np.all(done), pettingzoo env will not provide obs for env after done
            length_list.append(tag_interval_length)
            states =  env.reset()  # p1_state=p2_state
            for i in range(num_agents):
                reward_list[i].append(episode_reward[i])
                for j in range(args.num_envs):
                    writer.add_scalar(f"env{j} p{i}/episode_reward", episode_reward_separate[j][i], frame_idx*args.num_envs)

            writer.add_scalar("data/tag_interval_length", tag_interval_length, frame_idx*args.num_envs)
            tag_interval_length = 0
            episode_reward = [0 for _ in range(num_agents)]
            episode_reward_separate = [[0 for _ in range(num_agents)] for _ in range(args.num_envs)]

        if frame_idx % args.train_freq == 0:
            if (len(agent.replay_buffer) > args.rl_start):
                # Update Best Response with Reinforcement Learning
                rl_loss, qs = compute_rl_loss(agent, args)
                rl_loss_list.append(rl_loss.item())
                q_list.append(qs.item())

                if frame_idx % args.max_tag_interval == 0:  # not log at every step
                    writer.add_scalar(f"p{i}/rl_loss", rl_loss.item(), frame_idx*args.num_envs)

        if frame_idx % args.update_target == 0:
            update_target(agent.current_model, agent.target_model)

        # Logging and Saving models
        if frame_idx % args.evaluation_interval == 0:
            print(f"Frame: {frame_idx*args.num_envs}, Avg. RL Loss: {np.mean(rl_loss_list):.3f}, Avg. Q value: {np.mean(q_list):.3f}, Avg. Length: {np.mean(length_list):.1f}"+\
                ''.join([f", P{i} Avg. Reward: {np.mean(reward_list[i]):.3f}" for i in range(num_agents)]))
            reward_list = [[] for _ in range(num_agents)]
            rl_loss_list = []
            q_list = []
            length_list.clear()

            # agent.save_model(model_path+f'/{frame_idx}_')
            agent.save_model(model_path)
            # Evaluate the model, Output one Q table
            # q = agent.current_model(torch.FloatTensor([states[0].reshape(-1)]).to(args.device)).detach().cpu().numpy()
            # print('Q table: \n', q.reshape(env.action_space[0].n, -1))
            # dist = agent.compute_nash(np.array([q]), return_dist=True)
            # print('Nash policies: ', dist)

        # Render if rendering argument is on
        if args.render:
            env.render()
        t3=time.time()
        # print((t2-t1)/(t3-t1))

    agent.save_model(model_path+f'/{frame_idx}_')


def compute_rl_loss(agent, args):
    current_model, target_model, replay_buffer, optimizer = agent.current_model, agent.target_model, agent.replay_buffer, agent.rl_optimizer
    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
    weights = torch.ones(args.batch_size)
    # print(state.shape)
    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    # action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    # Q-Learning with target network
    q_values = current_model(state)
    # target_next_q_values_ = target_model(next_state)
    target_next_q_values_ = current_model(next_state)  # target model causing inaccuracy in Q estimation
    target_next_q_values = target_next_q_values_.detach().cpu().numpy()
    # print(q_values.shape)

    action_dim = int(np.sqrt(q_values.shape[-1])) # for two-symmetric-agent case only
    action = torch.LongTensor([a[0]*action_dim+a[1] for a in action]).to(args.device)
    # print(action.shape)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    # next_q_value = target_next_q_values.max(1)[0]  # original one, get the maximum of target Q

    # compute CCE or NE
    if args.cce: # Coarse Correlated Equilibrium
        cce_dists = agent.compute_cce(target_next_q_values, return_dist=True)
        target_next_q_values_ = target_next_q_values_.reshape(-1, action_dim, action_dim)
        cce_dists_  = torch.FloatTensor(cce_dists).to(args.device)
        next_q_value = torch.einsum('bij,bij->b', cce_dists_, target_next_q_values_)
        # print('value: ', reward.shape, next_q_value)

    else: # Nash Equilibrium
        nash_dists = agent.compute_nash(target_next_q_values, return_dist=True)  # get the mixed strategy Nash rather than specific actions
        target_next_q_values_ = target_next_q_values_.reshape(-1, action_dim, action_dim)
        nash_dists_  = torch.FloatTensor(nash_dists).to(args.device)
        next_q_value = torch.einsum('bk,bk->b', torch.einsum('bj,bjk->bk', nash_dists_[:, 0], target_next_q_values_), nash_dists_[:, 1])
        # print(next_q_value, target_next_q_values_)
        
        # next_q_value = torch.zeros_like(q_value) # test for rock-paper-scissor (stage game)

        # greedy Q estimation (cause overestimation, increasing Q value)
        # next_q_value = torch.FloatTensor(target_next_q_values).to(args.device)
        # next_q_value = torch.max(next_q_value, dim=-1)[0]

        # softmax prob average
        # softmax = torch.nn.Softmax(dim=-1)
        # next_q_value = torch.FloatTensor(target_next_q_values).to(args.device)
        # prob = softmax(next_q_value)
        # next_q_value = torch.sum(next_q_value*prob, dim=-1)

    expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

    # Huber Loss
    loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
    loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, torch.mean(q_value)  # return the q value to see whether overestimation

def test(env, args, model_path, num_agents=2): 
    agent = ParallelNashAgent(env, args)
    # find the latest model
    arr = os.listdir(model_path)
    arrv = []
    for f in arr:
        arrv.append(int(f.split('_')[0]))
    last_model_idx = max(arrv)
    model_path += f'{last_model_idx}_'
    agent.load_model(model_path, eval=True, map_location='cuda:0')  

    print('Load model from: ', model_path)

    reward_list = [[] for _ in range(num_agents)]
    length_list = []
    [n0, n1] = env.agents # agents name in one env, 2-player game

    for i in range(3):
        print('Episode: ', i)
        states = env.reset()
        episode_reward = [0 for _ in range(num_agents)]
        episode_length = 0
        t = 0
        while True:
            if args.render:
                env.render() 
            actions_ = agent.act(torch.FloatTensor(states).reshape(-1).unsqueeze(0), 0.)  # epsilon=0, greedy action
            assert num_agents == 2
            actions = [{n0: a0, n1: a1} for a0, a1 in zip(*actions_.T)][0]
            next_states, reward, done, _ = env.step(actions)

            states = next_states
            for i in range(num_agents):
                episode_reward[i] += reward[i]
            episode_length += 1
            if done or t>=args.max_tag_interval:  # the pong game might get stuck after a while: https://github.com/PettingZoo-Team/PettingZoo/issues/357
                for i in range(num_agents):
                    reward_list[i].append(episode_reward[i])
                length_list.append(episode_length)
                break
            t += 1
    print("Test Result - Length {:.2f} ".format(np.mean(length_list))+f'First Player Reward {np.mean(reward_list[i]):.2f}')

def main():
    args = get_args()
    print_args(args)
    model_path = f'models/nash_dqn/{args.env}/{args.save_model}'
    os.makedirs(model_path, exist_ok=True)

    log_dir = create_log_dir(args)
    if not args.evaluate:
        writer = SummaryWriter(log_dir)
    SEED = 721
    if args.num_envs == 1 or args.evaluate:
        env = make_env(args)  # "SlimeVolley-v0", "SlimeVolleyPixel-v0" 'Pong-ram-v0'
    else:
        VectorEnv = [DummyVectorEnv, SubprocVectorEnv][1]  # https://github.com/thu-ml/tianshou/blob/master/tianshou/env/venvs.py
        env = VectorEnv([lambda: make_env(args) for _ in range(args.num_envs)])

    print(env.observation_space, env.action_space)

    set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.evaluate:
        model_path = f'models/nash_dqn/{args.env}/{args.load_model}/'
        test(env, args, model_path)
        env.close()
        return

    train(env, args, writer, model_path)

    # writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
