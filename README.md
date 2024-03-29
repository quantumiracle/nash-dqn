# Nash-DQN

This repository provides a standalone version of **Nash-DQN** and **Nash-DQN-with-Exploiter** algorithm for two-player zero-sum Markov games, details see [**our paper**: A Deep Reinforcement Learning Approach for Finding Non-Exploitable Strategies in Two-Player Atari Games. Zihan Ding, Dijia Su, Qinghua Liu, Chi Jin](https://arxiv.org/abs/2207.08894). 

The **Nash-DQN** and **Nash-DQN-with-Exploiter** algorithms are also compared against other baselines methods like Self-play, Fictitious Self-play, Neural Fictitious Self-play, Policy Space Response Oracle, but in another library called [**MARS**](https://github.com/quantumiracle/MARS).

## File Structure

* `equilibrium_solver/` contains different solvers for Nash equilibrium, including [ECOS](https://github.com/embotech/ecos), [Nashpy](https://github.com/drvinceknight/Nashpy), [PuLP](https://github.com/coin-or/pulp), [CVXPY](https://github.com/cvxpy/cvxpy), [Gurobipy](https://www.gurobi.com/), etc.
* `solver_comparison.ipynb` provides a time and solvability analysis for all solvers implemented in this repo;
* `common/` contains all necessary RL components, including networks, env wrappers, buffers, training arguments, etc;
* `agent.py` is the base class of learning agents;
* `dqn.py` is the implementation of the DQN algorithm that Nash-DQN inherits;
* `nash_dqn.py` is the implementation of the Nash-DQN algorithm;
* `nash_dqn_exploiter.py` is the implementation of the Nash-DQN-with-Exploiter algorithm;
* `launch.py` is the launching script for train/test the algorithm;
* `confs/default.yaml` is the default configurations;
* `confs/pettingzoo_boxing_v2_nash_dqn.yaml` is an example configuration script for PettingZoo Boxing-v2 with Nash-DQN algorithm. Note that it will overwrite the same entries in the default configurations;
* `confs/pettingzoo_boxing_v2_nash_dqn_exploiter.yaml` is an example configuration script for PettingZoo Boxing-v2 with Nash-DQN-with-Exploiter algorithm. Note that it will overwrite the same entries in the default configurations;




## Requirements

* Using Conda environment:
```bash
git clone https://github.com/quantumiracle/nash-dqn.git && cd nash-dqn
pip install -r requirements.txt
AutoROM --accept-license
```

The ECOS solver is needed, which can be installed via `pip install ecos`.

Others are general ML packages, like torch, tensorboard, numpy, gym, pettingzoo, supersuit, etc.

* Using [Poetry](https://python-poetry.org/):
```
git clone https://github.com/quantumiracle/nash-dqn.git && cd nash-dqn
poetry install 
poetry run AutoROM --accept-license
poetry run pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
poetry run python launch.py --env pettingzoo_boxing_v2 --method nash_dqn
```

## Quick Start

To train Nash-DQN/Nash-DQN-with-Exploiter on two-agent zero-sum game, *boxing-v2* in Pettingzoo, run:

```bash
python launch.py --env pettingzoo_boxing_v2 --method nash_dqn
python launch.py --env pettingzoo_boxing_v2 --method nash_dqn_exploiter

```

Train with [Wandb](https://wandb.ai) (for logging training results):

```
python launch.py --env pettingzoo_boxing_v2 --method nash_dqn --wandb_activate True --wandb_entity **YourWandbAccountName** --wandb_project **ProjectName**
```

Test after training:

```bash
python launch.py --env pettingzoo_boxing_v2 --method nash_dqn --test True --load_model_idx './model/boxing_v2_NashDQN_2022-07-26-17-42-20/2000' --render True
```

## Detailed Instructions

Full list of configurations:

    Environment args:
    * '--env', type=str, default=None, help='environment type and name'
    * '--num_envs', type=int, default=1, help='number of environments for parallel sampling'
    * '--ram', type=bool, default=False, help='use RAM observation'
    * '--render', type=bool, default=False, help='render the scene'
    * '--seed', type=str, default='random', help='random seed'
    * '--record_video', type=bool, default=False, help='whether recording the video'

    Agent args:
    * '--algorithm', type=str, default=None, help='algorithm name'
    * '--algorithm_spec.dueling', type=bool, default=False, help='DQN: dueling trick'
    * '--algorithm_spec.replay_buffer_size', type=int, default=1e5, help='DQN: replay buffer size'
    * '--algorithm_spec.gamma', type=float, default=0.99, help='DQN: discount factor'
    * '--algorithm_spec.multi_step', type=int, default=1, help='DQN: multi-step return'
    * '--algorithm_spec.target_update_interval', type=bool, default=False, help='DQN: steps skipped for target network update'
    * '--algorithm_spec.eps_start', type=float, default=1, help='DQN: epsilon-greedy starting value'
    * '--algorithm_spec.eps_final', type=float, default=0.001, help='DQN: epsilon-greedy ending value'
    * '--algorithm_spec.eps_decay', type=float, default=5000000, help='DQN: epsilon-greedy decay interval'

    Training args:
    * '--num_process', type=int, default=1, help='use multiprocessing for both sampling and update'
    * '--batch_size', type=int, default=128, help='batch size for update'
    * '--max_episodes', type=int, default=50000, help='maximum episodes for rollout'
    * '--max_steps_per_episode', type=int, default=300, help='maximum steps per episode'
    * '--train_start_frame', type=int, default=0, help='start frame for training (not update when warmup)'
    * '--method', dest='marl_method', type=str, default=None, help='method name'
    * '--save_id', type=str, default='0', help='identification number for each run'
    * '--optimizer', type=str, default='adam', help='choose optimizer'
    * '--batch_size', type=int, default=128, help='batch size for update'
    * '--learning_rate', type=float, default=1e-4, help='learning rate'
    * '--device', type=str, default='gpu', help='computation device for model optimization'
    * '--update_itr', type=int, default=1, help='number of updates per step'
    * '--log_avg_window', type=int, default=20, help='window length for averaging the logged results'
    * '--log_interval', type=int, default=20, help='interval for logging'
    * '--test', type=bool, default=False, help='whether in test mode'
    * '--exploit', type=bool, default=False, help='whether in exploitation mode'
    * '--load_model_idx', type=str, default='0', help='index of the model to load'
    * '--load_model_full_path', type=str, default='/', help='full path of the model to load'
    * '--multiprocess', type=bool, default=False, help='whether using multiprocess or not'
    * '--eval_models', type=bool, default=False, help='evalutation models during training (only for specific methods)'
    * '--save_path', type=str, default='/', help='path to save models and logs'
    * '--save_interval', type=int, default=2000, help='episode interval to save models'
    * '--wandb_activate', type=bool, default=False, help='activate wandb for logging'
    * '--wandb_entity', type=str, default='', help='wandb entity'
    * '--wandb_project', type=str, default='', help='wandb project'
    * '--wandb_group', type=str, default='', help='wandb project'
    * '--wandb_name', type=str, default='', help='wandb name'
    * '--net_architecture.hidden_dim_list', type=str, default='[128, 128, 128]', help='list of hidden dimensions for model'
    * '--net_architecture.hidden_activation', type=str, default='ReLU', help='hidden activation function'
    * '--net_architecture.output_activation', type=str, default=False, help='output activation function'
    * '--net_architecture.hidden_activation', type=str, default='ReLU', help='hidden activation function'
    * '--net_architecture.channel_list', type=str, default='[8, 8, 16]', help='list of channels for CNN'
    * '--net_architecture.kernel_size_list', type=str, default='[4, 4, 4]', help='list of kernel sizes for CNN'
    * '--net_architecture.stride_list', type=str, default='[2, 1, 1]', help='list of strides for CNN'
    * '--marl_spec.min_update_interval', type=int, default=20, help='mininal opponent update interval in unit of episodes'
    * '--marl_spec.score_avg_window', type=int, default=10, help='the length of window for averaging the score values'
    * '--marl_spec.global_state', type=bool, default=False, help='whether using global observation'
    
   ## Citation
   ```
   @article{ding2022deep,
  title={A Deep Reinforcement Learning Approach for Finding Non-Exploitable Strategies in Two-Player Atari Games},
  author={Ding, Zihan and Su, Dijia and Liu, Qinghua and Jin, Chi},
  journal={arXiv preprint arXiv:2207.08894},
  year={2022}
}

   ```
