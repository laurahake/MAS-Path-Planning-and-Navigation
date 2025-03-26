from custom_environment import raw_env
import time
import numpy as np
from simple_pid import PID
from scipy.special import softmax
import pickle
import random

def print_state(state, size=9):
    print("The state is:")
    for c in range(size-1, -1, -1):
        column = state[c::size] # every sizeth element
        print(" ".join(f"{cell:2}" for cell in column))

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = []
        self.new_state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.terminal_memory = []
        
    def store_transition(self, state, action, reward, next_state, terminate):
        
        if self.mem_counter < self.mem_size:
            self.state_memory.append(state)
            self.new_state_memory.append(next_state)
            self.action_memory.append(action)
            self.reward_memory.append(reward)
            self.terminal_memory.append(terminate)
        else:
            # Wenn der Speicher voll ist, ersetze ältere Einträge (FIFO-Prinzip)
            index = self.mem_counter % self.mem_size
            self.state_memory[index] = state
            self.new_state_memory[index] = next_state
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.terminal_memory[index] = terminate

        self.mem_counter += 1 
        
    def sample_buffer(self, batchsize):
        
        max_mem = min(self.mem_counter, self.mem_size) 
        batch = np.random.choice(max_mem, batchsize, replace=False)
        
        states = [self.state_memory[i] for i in batch]
        next_states = [self.new_state_memory[i] for i in batch]
        actions = [self.action_memory[i] for i in batch]
        rewards = [self.reward_memory[i] for i in batch]
        terminates = [self.terminal_memory[i] for i in batch]
        
        return states, actions, rewards, next_states, terminates
    

def train(seed = None, kappa=1, T=10000, N=10, batchsize = 32, p = 0, c=1):
    
    if seed is not None:
        np.random.seed(seed)
    
    env = raw_env(render_mode="human")
    agent_states, agent_observations = env.reset(seed=42)
    
    action_space_num = 5
    gamma = 0.95
    
    replay_buffer = ReplayBuffer(max_size=50000, state_dim = 1, action_dim = 1)
    Q={}
    
    def epsilon_greedy_policy(state, Q, episode, epsilon_min = 0.1, epsilon_decay = 0.999):
        if episode <= 10:
            epsilon = 1
        else:
            epsilon = max(epsilon_min, epsilon_decay ** (episode - 10)) 

        if random.uniform(0, 1) < epsilon:
            return random.choice(range(action_space_num))
        else:
            if state not in Q:
                Q[state] = np.zeros(action_space_num)
            return np.argmax(Q[state])

    
    TD_error_per_episode = []
    reward_per_episode = []
    episode_length = 100
    TD_error_episode = 0
    train = False
    stepsize = [c/((step/N)**(p) + 1) for step in range(T)]
    agent_done = {agent: False for agent in env.agents}
    episode = 1
    
    for step in range(T):
        agent_actions = {}
        
        env.agent_selection = env.agents[0]
        for agent in env.agents:
            observation = agent_observations[agent]
            if env.terminations[agent]:
                action = None
            else:
                discrete_action = epsilon_greedy_policy(agent_states[agent], Q, episode)
                action = env.get_cont_action(observation, env.world.dim_p, discrete_action, agent)
                agent_actions[agent] = discrete_action
            env.step(action)
        
        env.agent_selection = env.agents[0]
        for agent in env.agents:
            if agent_done[agent] == True:
                env.next_agent()
                continue
            
            observation, reward, termination, truncation, info, next_state = env.last()
            agent_observations[agent] = observation 
            
            #print(agent + ": Erreicht wurde Punkt: " + str(observation[0]) + ", " + str(observation[1]))
            if termination or truncation:
                agent_done[agent] = True
            replay_buffer.store_transition(agent_states[agent], agent_actions[agent], reward,next_state,agent_done[agent])
            agent_states[agent] = next_state
            
            if not train:
                if replay_buffer.mem_counter > 2*batchsize:
                    train = True
            if train:
                states, actions, rewards, next_states, terminates = replay_buffer.sample_buffer(batchsize)
                for i in range(batchsize):
                    s, a, r, s_next, term = states[i], actions[i], rewards[i], next_states[i], terminates[i]
                    if s not in Q:
                        Q[s] = np.zeros(action_space_num)
                    if s_next not in Q:
                        Q[s_next] = np.zeros(action_space_num)
                
                    temporal_difference_error = Q[s][a] - (r + gamma * np.max(Q[s_next]) * (1 - term))
                    TD_error_episode += np.square(temporal_difference_error)
                    Q[s][a] -= stepsize[step] * temporal_difference_error
            
            if all(agent_done[agent] for agent in agent_done):
                agent_states, agent_observations = env.reset()
                episode +=1
                agent_done = {agent: False for agent in env.agents}
                break
        
            else:   
                env.next_agent()


        if step%episode_length == 0 and step != 0 and episode>10:
            TD_error_per_episode.append(TD_error_episode/episode_length)
            TD_error_episode = 0
            
            # Evaluation der aktuellen greedy policy
            reward_episode = 0
            evaluation_steps = 0
            K = 10
            for _ in range(K):
                agent_states, agent_observations = env.reset(seed=seed)
                agent_done = {agent: False for agent in env.agents}

                while not all(agent_done[agent] for agent in agent_done):
                    env.agent_selection = env.agents[0]
                    for agent in env.agents:
                        if env.terminations[agent]:
                            action = None
                        else:
                            state = agent_states[agent]
                            observation = agent_observations[agent]
                            if state not in Q:
                                Q[state] = np.zeros(action_space_num)
                            discrete_action = np.argmax(Q[state])
                            action = env.get_cont_action(observation, env.world.dim_p, discrete_action, agent)
                        env.step(action)
                        
                    env.agent_selection = env.agents[0]
                    for agent in env.agents:
                        if agent_done[agent] == True:
                            env.next_agent()
                            continue
                        observation, reward, termination, truncation, info, state = env.last()
                        agent_observations[agent] = observation
                        if termination or truncation:
                            agent_done[agent] = True
                        agent_states[agent] = state
                        reward_episode += reward
                        evaluation_steps += 1
                        env.next_agent()
                    
            reward_per_episode.append(reward_episode/evaluation_steps)
            
            agent_states, agent_observations= env.reset()
            agent_done = {agent: False for agent in env.agents}
                
        print(str(step))

    env.close()
    return Q, TD_error_per_episode, reward_per_episode
        
Q, TD_error_per_episode, reward_per_episode =train()
with open("training_data.pkl", "wb") as f:
    pickle.dump((Q, TD_error_per_episode, reward_per_episode), f)

print("Trainingsergebnisse gespeichert.")