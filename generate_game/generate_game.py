from gym.spaces import Discrete
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

def GenerateGeneralSumMatrixGame(matrix, name):
    """ 
    Modified from prettingzoo rps.py
    Generate a general-sum bimatrix game like rock-paper-scissor.
    """
    assert isinstance(matrix, np.ndarray)
    if len(matrix.shape) > 2: # bimatrix
        row, col = matrix.shape[1:]
        BIMATRIX = True
    else:   # unimatrix, zero-sum
        row, col = matrix.shape
        BIMATRIX = False

    assert row ==  col
    MOVES = [chr(65+i) for i in range(row)]  # ['A', 'B', ...]
    MOVES.append('None')
    print(MOVES)
    for i in range(row):
        exec(chr(65+i)+f'={i}', globals())  # default as locals() will not work
    NONE = row
    NUM_ITERS = 100
    

    class raw_env(AECEnv):
        """Two-player environment for rock paper scissors.
        The observation is simply the last opponent action."""

        metadata = {'render.modes': ['human'], }

        def __init__(self,):
            self.agents = ["player_" + str(r) for r in range(2)]
            self.possible_agents = self.agents[:]
            self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
            self.action_spaces = {agent: Discrete(row) for agent in self.agents}
            self.observation_spaces = {agent: Discrete(row+1) for agent in self.agents}
            self.metadata ["name"] = name
            self.reinit()

        def reinit(self):
            self.agents = self.possible_agents[:]
            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.next()
            self.rewards = {agent: 0 for agent in self.agents}
            self._cumulative_rewards = {agent: 0 for agent in self.agents}
            self.dones = {agent: False for agent in self.agents}
            self.infos = {agent: {} for agent in self.agents}
            self.state = {agent: NONE for agent in self.agents}
            self.observations = {agent: NONE for agent in self.agents}
            self.num_moves = 0

        def render(self, mode="human"):
            string = ("Current state: Agent1: {} , Agent2: {}".format(MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]))
            print(string)
            return string

        def observe(self, agent):
            # observation of one agent is the previous state of the other
            return np.array(self.observations[agent])

        def close(self):
            pass

        def reset(self):
            self.reinit()

        def step(self, action):
            if self.dones[self.agent_selection]:
                return self._was_done_step(action)
            agent = self.agent_selection

            self.state[self.agent_selection] = action

            # collect reward if it is the last agent to act
            if self._agent_selector.is_last():
                # self.rewards[self.agents[0]], self.rewards[self.agents[1]] = {
                #     (ROCK, ROCK): (0, 0),
                #     (ROCK, PAPER): (-1, 1),
                #     (ROCK, SCISSORS): (1, -1),
                #     (PAPER, ROCK): (1, -1),
                #     (PAPER, PAPER): (0, 0),
                #     (PAPER, SCISSORS): (-1, 1),
                #     (SCISSORS, ROCK): (-1, 1),
                #     (SCISSORS, PAPER): (1, -1),
                #     (SCISSORS, SCISSORS): (0, 0),
                # }[(self.state[self.agents[0]], self.state[self.agents[1]])]
                payoff_list=[]
                if BIMATRIX:
                    for i in range(row):
                        for j in range(row):
                            payoff_list.append(f"({MOVES[i]}, {MOVES[j]}): ({matrix[0][i][j]}, {matrix[1][i][j]})")
                else:
                    for i in range(row):
                        for j in range(row):
                            payoff_list.append(f"({MOVES[i]}, {MOVES[j]}): ({matrix[i][j]}, -{matrix[i][j]})")
                exec("self.rewards[self.agents[0]], self.rewards[self.agents[1]] = {"+','.join(payoff_list)+"}[(self.state[self.agents[0]], self.state[self.agents[1]])]")

                self.num_moves += 1
                self.dones = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}

                # observe the current state
                for i in self.agents:
                    self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
            else:
                self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
                self._clear_rewards()

            self._cumulative_rewards[self.agent_selection] = 0
            self.agent_selection = self._agent_selector.next()
            self._accumulate_rewards()

    def env():
        env = raw_env()
        env = wrappers.CaptureStdoutWrapper(env)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    parallel_env = parallel_wrapper_fn(env)
    return parallel_env()



if __name__ == "__main__":
    payoff_matrix = np.array([[0, -1, 1], [2, 0, -1], [-1, 1, 0]])
    payoff_bimatrix = np.array([[[10, 0], [3, 2]],  [[9, 3], [0, 2]]])
    env = GenerateGeneralSumMatrixGame(payoff_matrix, 'random')
    print(env)
