import UUVEnv
import numpy as np
import fimdp

grid_size = [20,20]
num_agents = 3
capacities = [100 for i in range(num_agents)]
num_actions = 8
num_states = grid_size[0]*grid_size[1]
init_states = [np.random.choice(num_states)  for i in range(num_agents)]
reloads = list(np.random.choice(num_states, num_states//10))
targets = list(np.random.choice(num_states, num_states//10))


def test_uuv_structure():
    env = UUVEnv.SynchronousMultiAgentEnv(num_agents, grid_size, capacities, reloads, targets, init_states)
    assert env.num_agents == num_agents
    assert env.num_states == num_states
    assert env.num_actions == num_actions
    assert env.init_states == init_states
    assert env.positions == init_states
    assert env.capacities == capacities
    assert env.energies == capacities
    assert env.states == [s for s in range(num_states)] 
    assert env.reloads == reloads
    assert env.targets == targets
    assert env.num_timesteps == 0
    assert env.state_histories == [[s] for s in init_states]
  
def test_uuv_transprob():
    env = UUVEnv.SynchronousMultiAgentEnv(num_agents, grid_size, capacities, reloads, targets, init_states)
    assert np.shape(env.trans_prob) == (num_states, num_actions, num_states)
    
    for s in range(num_states):
        for a in range(num_actions):
            next_state = np.random.choice(num_states, p=env.trans_prob[s, a, :])
            assert next_state in env.states
            
    
def test_uuv_cmdp():
    env = UUVEnv.SynchronousMultiAgentEnv(num_agents, grid_size, capacities, reloads, targets, init_states)
    cmdp, targets_list = env.get_consmdp()
    assert targets_list == targets
    assert type(cmdp) == fimdp.core.ConsMDP
    assert type(env.consmdp) == fimdp.core.ConsMDP
    

    
    

