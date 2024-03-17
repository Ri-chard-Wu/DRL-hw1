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

    def __init__(self, 
                 t=0,
                 discount_factor=0.99):

        self.update_parameters(t)  # init explore rate and learning rate
        
        self.discount_factor = discount_factor
        self.nA = 6
        self.nS = 500
        self.q_table = np.zeros((self.nS, self.nA))

    
    def select_action(self, state): 
        if np.random.rand() < self.exploring_rate:
            action = np.random.choice(self.nA)  # Select a random action
        else:
            action = np.argmax(self.q_table[state])  # Select the action with the highest q
        return action

    def update_policy(self, state, action, reward, state_prime): 
        best_q = np.max(self.q_table[state_prime])
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


 
agent = Agent()
 
 
reward_per_epoch = []
lifetime_per_epoch = []
exploring_rates = []
learning_rates = []
print_every_episode = 500
show_gif_every_episode = 5000
NUM_EPISODE = 40000
for episode in range(0, NUM_EPISODE):
 
    observation, info = env.reset(seed=42) 

    # for every 500 episodes, shutdown exploration to see performance of greedy action
    if episode % print_every_episode == 0:
        agent.shutdown_explore()
 
    cum_reward = 0  
    t = 0

    while(1): 
        action = agent.select_action(observation) 
        observation_next, reward, terminated, truncated, info = env.step(action) 
        cum_reward += reward
 
        agent.update_policy(observation, action, reward, observation_next)
 
        observation = observation_next
        t += 1

        if terminated or truncated: break
 
    agent.update_parameters(episode)

    if episode % print_every_episode == 0:
        print("Episode {} finished after {} time steps, cumulated reward: {}, exploring rate: {}, learning rate: {}".format(
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

    # # for every 5000 episode, record an animation
    # if episode % show_gif_every_episode == 0:
    #     print("len frames:", len(frames))
    #     clip = make_anim(frames, fps=60, true_image=True).rotate(-90)
    #     display(clip.ipython_display(fps=60, autoplay=1, loop=1))


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
 