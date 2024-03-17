

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
 
 