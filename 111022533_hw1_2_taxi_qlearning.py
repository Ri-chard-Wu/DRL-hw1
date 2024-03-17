import gymnasium as gym
import numpy as np
import torch


MIN_EXPLORING_RATE = 0.01
MAX_EXPLORING_RATE = 0.5
MIN_LEARNING_RATE = 0.5
MAX_LEARNING_RATE = 0.5

# https://www.gymlibrary.dev/environments/toy_text/taxi/
env = gym.make('Taxi-v3') 
env.action_space.seed()


nA = 6
nS = 500

 

class Agent:

    def __init__(self, nA, nS, 
                 t=0,
                 discount_factor=0.99):

        self.update_parameters(t)  # init explore rate and learning rate
        
        self.discount_factor = discount_factor
        self.nA = nA
        self.nS = nS
        # self.q_table = np.zeros((self.nS, self.nA))
        self.q_table = torch.zeros([self.nS, self.nA], dtype=torch.float32)

    
    def select_action(self, state): 
        if np.random.rand() < self.exploring_rate:
            action = np.random.choice(self.nA)  # Select a random action
        else:
            # action = np.argmax(self.q_table[state])  # Select the action with the highest q
            action = torch.argmax(self.q_table[state]).item()
        return action

    def update_policy(self, state, action, reward, state_prime): 
        best_q = torch.max(self.q_table[state_prime])
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * best_q - self.q_table[state][action])

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



renderer = Renderer(nA, nS, env)

def train(save_path=None):

    agent = Agent(nA, nS)
    
    
    reward_per_epoch = []
    lifetime_per_epoch = []
    exploring_rates = []
    learning_rates = []
    print_every_episode = 1
    show_gif_every_episode = 5000
    NUM_EPISODE = 2500
    for episode in range(0, NUM_EPISODE):
    
        observation, info = env.reset() 

        # for every 500 episodes, shutdown exploration to see performance of greedy action
        if episode % print_every_episode == 0:
            agent.shutdown_explore()
    
        cum_reward = 0  
        t = 0
        s_a_pairs = []

        while(1): 
            action = agent.select_action(observation) 
            observation_next, reward, terminated, truncated, info = env.step(action) 
            cum_reward += reward
        
            agent.update_policy(observation, action, reward, observation_next)
    
            s_a_pairs.append([observation, action])
            observation = observation_next
            t += 1

            if terminated or truncated: 
                s_a_pairs.append([observation, None])
                # print(f'done reward: {reward}')
                break
    
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

        # for every 5000 episode, record an animation
        if episode == NUM_EPISODE-1:
            # print("len frames:", len(frames))
            # clip = make_anim(frames, fps=60, true_image=True).rotate(-90)
            # display(clip.ipython_display(fps=60, autoplay=1, loop=1))
            renderer.render_all(s_a_pairs)

    print(f'reward_per_epoch (qlearning): {reward_per_epoch}')
    if(save_path is not None):
        print('save model')
        torch.save(agent.q_table, save_path)
      



# train('111022533_hw1_2_taxi_qlearning.pth')
# train()





def evaluate(path):

    # agent = Agent(nA, nS)
    # agent.load_table(path)
    # agent.shutdown_explore()

    agent = torch.load(path)
    # print(agent)
    cum_reward_avg = 0
    n = 10
    for i in range(n):
        
        observation, info = env.reset() 

        cum_reward = 0          
        t = 0
      
        while(1):  

            action = torch.argmax(agent[observation]).item()
            
            observation_next, reward, terminated, truncated, info = env.step(action) 
            
            cum_reward += reward

            if terminated or truncated: 
                         
                break                
        
            t += 1
            observation = observation_next

        cum_reward_avg += cum_reward
        print(f'cum_reward: {cum_reward}, t: {t}')
    cum_reward_avg = cum_reward_avg / n
    print(f'cum_reward_avg: {cum_reward_avg}')

evaluate('111022533_hw1_2_taxi_qlearning.pth')
 
 