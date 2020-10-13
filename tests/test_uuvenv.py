from fimdpenv import UUVEnv
import numpy as np
import fimdp



def test_uuv_structure():
    env = UUVEnv.SingleAgentEnv([20,20], 40, [1,3,5], [11,33])
    assert env.num_states == 400
    assert env.num_actions == 8
    
    
def test_uuv_strategy():
    env = UUVEnv.SingleAgentEnv([20,20], 40, [1,3,5], [11,33])
    env.create_consmdp()
    assert type(env.consmdp) == fimdp.consMDP.ConsMDP
    
def test_uuv_history():
    env = UUVEnv.SingleAgentEnv([20,20], 40, [1,3,5], [11,33], init_state=20)
    info = env.step(4)
    info = env.step(4)
    
    assert env.state_history == [env.init_state, env.init_state+1, env.init_state+2]
    assert env.action_history == [4,4]
    assert env.energy == env.capacity - 2*env.saction_cost
    
    

