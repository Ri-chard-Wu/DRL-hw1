import gymnasium as gym
import numpy as np
# # import gym
# env = gym.make("LunarLander-v2", render_mode="rgb_array")
# env.action_space.seed(42)

# observation, info = env.reset(seed=42)

# for _ in range(1000):
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()
MIN_EXPLORING_RATE = 0.01
MAX_EXPLORING_RATE = 0.5
MIN_LEARNING_RATE = 0.5
MAX_LEARNING_RATE = 0.5

# https://www.gymlibrary.dev/environments/toy_text/taxi/
env = gym.make('Taxi-v3') 
env.action_space.seed(42)



 

class Agent:

    def __init__(self, nA, nS, 
                 t=0,
                 discount_factor=0.99):

        self.update_parameters(t)  # init explore rate and learning rate
        
        self.discount_factor = discount_factor
        self.nA = nA
        self.nS = nS
        self.q_table = np.zeros((self.nS, self.nA))

    
    def select_action(self, state): 
        if np.random.rand() < self.exploring_rate:
            action = np.random.choice(self.nA)  # Select a random action
        else:
            action = np.argmax(self.q_table[state])  # Select the action with the highest q
        return action

    def update_policy(self, state, action, reward, state_prime, action_prime): 
        # best_q = np.max(self.q_table[state_prime])
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * self.q_table[state_prime][action_prime] - self.q_table[state][action])

    def update_parameters(self, episode):
        self.exploring_rate = \
                max(MIN_EXPLORING_RATE, min(MAX_EXPLORING_RATE, 0.99**((episode) / 30)))
        self.learning_rate = \
                max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, 0.99 ** ((episode) / 30)))

    def shutdown_explore(self):
        # make action selection greedy
        self.exploring_rate = 0


 
class Renderer:

    def __init__(self, nA, nS, env):
        self.env = env
        self.nA = nA
        self.nS = nS    
        self._map = ["+---------+",
                     "|R: | : :G|",
                     "| : | : : |",
                     "| : : : : |",
                     "| | : | : |",
                     "|Y| : |B: |",
                     "+---------+"]
        self.map = []

        self.locMap = {
            0: (0, 0),
            1: (0, 4),
            2: (4, 0),
            3: (4, 3)
        }

        self.actDecode = {
            0: 'down',
            1: 'up',
            2: 'right',
            3: 'left',
            4: 'pick up',
            5: 'drop off' 
        }

    def get_pos(self, locIdx):
        """ locIdx
            0: R(ed)
            1: G(reen)
            2: Y(ellow)
            3: B(lue)
        """        
        if(locIdx in self.locMap):
            return self.locMap[locIdx]
        else:
            return None

    def init_map(self):
        self.map = [i for i in self._map]


    def place(self, sym, pos):
        if(pos is None): return
        r = pos[0]
        c = pos[1]
        r_str = self.map[1 + r]        
        r_char = [*r_str]
        r_char[int(c*2+1)] = sym
        self.map[1 + r] = ''.join(r_char)

    def render(self, sa):
        taxi_r, taxi_c, src, des = [i for i in self.env.unwrapped.decode(sa[0])]
        self.init_map()
        self.place('s', self.get_pos(src))
        self.place('d', self.get_pos(des))
        self.place('@', (taxi_r, taxi_c))
        
        print('\n'.join(self.map))
        print(f'action: {self.decodeAction(sa[1])}')
        print()

        
    
    def decodeAction(self, a):
        if (a is None): return None
        else: return self.actDecode[a]

    def render_all(self, sa_all):
        
        for sa in sa_all:
            self.render(sa)



nA = 6
nS = 500
renderer = Renderer(nA, nS, env)
agent = Agent(nA, nS)
 
 
reward_per_epoch = []
lifetime_per_epoch = []
exploring_rates = []
learning_rates = []
print_every_episode = 1
show_gif_every_episode = 5000
NUM_EPISODE = 100
for episode in range(0, NUM_EPISODE):
 
    observation, info = env.reset(seed=42) 
    action = agent.select_action(observation) 
    
    if episode % print_every_episode == 0:
        agent.shutdown_explore()
 
    cum_reward = 0  
    t = 0
    s_a_pairs = [[observation, action]]

    while(1):  
        observation_next, reward, terminated, truncated, info = env.step(action) 
        
        cum_reward += reward

        if terminated or truncated: 
            s_a_pairs.append([observation_next, None])            
            break                
    
        action_next = agent.select_action(observation_next) 
        agent.update_policy(observation, action, reward, observation_next, action_next)
 
        observation = observation_next
        action = action_next
        s_a_pairs.append([observation, action])
        t += 1


 
    agent.update_parameters(episode)

    if episode % print_every_episode == 0:
        print("eps: {}, len: {}, cumu reward: {}, exploring rate: {}, learning rate: {}".format(
            episode,
            t,
            cum_reward,
            agent.exploring_rate,
            agent.learning_rate
        ))
        reward_per_epoch.append(cum_reward)
        exploring_rates.append(agent.exploring_rate)
        learning_rates.append(agent.learning_rate)
        lifetime_per_epoch.append(t)

    if episode == NUM_EPISODE-1:
        renderer.render_all(s_a_pairs)






print(agent.q_table)

# """
# print([i for i in env.unwrapped.decode(observation)]):

#     taxi_row, taxi_col, passenger_location, destination

#     Passenger locations:

#     0: R(ed)
#     1: G(reen)
#     2: Y(ellow)
#     3: B(lue)
#     4: in taxi

#     Destinations:

#     0: R(ed)
#     1: G(reen)
#     2: Y(ellow)
#     3: B(lue)
# """
# print([i for i in env.unwrapped.decode(observation)])


# print('`````````````````')
# print(info)
 