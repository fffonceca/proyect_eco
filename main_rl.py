from lib.util_rl import *
import numpy as np
import glfw
import gym

def main():
    test_steps = 1000000
    torques = np.array([0.00, 0.00])
    glfw.init()
    env = gym.make("Reacher-v2")  # brazo 2 DOF
    
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 256

    value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
    target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

    soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device) 
    policy_net = PolicyNetwork(state_dim, 
    action_dim, hidden_dim).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)


    value_criterion  = nn.MSELoss()
    soft_q_criterion1 = nn.MSELoss()
    soft_q_criterion2 = nn.MSELoss()
    lr  = 3e-4

    value_optimizer  = optim.Adam(value_net.parameters(), lr=lr)
    soft_q_optimizer = optim.Adam(soft_q_net1.parameters(), lr=lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    state, _ = env.reset(seed=None, return_info=True)
    replay_buffer_size = 100000
    replay_buffer = ReplayBuffer(replay_buffer_size)





    

    episode_reward = 0
    for _ in range(test_steps):
        env.render()
        torques = env.action_space.sample()
        next_state, reward, done, _ = env.step(torques)
        replay_buffer.push(state, torques, reward, next_state, done)
        state = next_state
        episode_reward += reward

        
    glfw.terminate()
    env.close()


if __name__ == '__main__':
    main()








#action_dim = env.action_space.shape[0]
#state_dim  = env.observation_space.shape[0]
#hidden_dim = 256
#
#value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
#target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)
#
#soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
#policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
#
#for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
#    target_param.data.copy_(param.data)
#    

