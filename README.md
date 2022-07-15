# Nash-DQN

## File Structure

* `equilibrium_solver/` contains different solvers for Nash equilibrium, including [ECOS](https://github.com/embotech/ecos), [Nashpy](https://github.com/drvinceknight/Nashpy), [PuLP](https://github.com/coin-or/pulp), [CVXPY](https://github.com/cvxpy/cvxpy), [Gurobipy](https://www.gurobi.com/), etc.
* `solver_comparison.ipynb` provides a time and solvability analysis for all solvers implemented in this repo;
* `common/` contains all necessary RL components, including networks, env wrappers, buffers, training arguments, etc;
* `nash_dqn.py` is the implementation of the Nash-DQN algorithm;
* `bilateral_dqn.py` is to train two agents (DQN) at the same time;



## Requirements

The ECOS solver is needed, which can be installed via:

```bash
pip install ecos
```

Others are general ML packages, like torch, tensorboard, numpy, gym, etc.



## Quick Start

To train Nash-DQN on two-agent zero-sum game, *boxing-v1* in Pettingzoo, run:

```bash
python nash_dqn.py --env boxing_v1 --num-envs 3 --ram --hidden-dim 256 --evaluation-interval 50000 --max-tag-interval 10000 --train-freq 100 --batch-size 1024 --max-frames 500000000
```

Test after training:

```bash
python nash_dqn.py --env boxing_v1 --num-envs 3 --ram --hidden-dim 256 --evaluate --render
```



## Usage Details

### Two Agents Nash DQN

For two agents zero-sum game with Nash DQN:

Test with rps_v1 (gamma is set 0 b.c. it is a repeated stage game):

   `python nash_dqn.py --env rps_v1 --num-envs 2 --hidden-dim 64 --evaluation-interval 500 --rl-start 1000 --lr 0.0001 --gamma 0`

   `python nash_dqn.py  --env SlimeVolley-v0 --hidden-dim 256 --num_envs 5 --max-tag-interval 3000`

   `python nash_dqn.py  --env pong_v1 --ram --hidden-dim 32 --num_envs 2 --max-tag-interval 10000` 

Note: 

* `pong_v1` is the `Pong` game from PettingZoo for two agents, need to specify `--ram` for RAM control, otherwise it is image-based control.
