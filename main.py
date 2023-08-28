from transmit_data_env import *
from dqn import *
from lib_ import *


if __name__ == "__main__":

    activate_GPU()

    EPISODES = 1000

    # Example usage
    num_user = 4
    number_power = 3
    max_power = 0.0316227766
    max_channel = 3

    env = IoTCommunicationEnv(num_user, number_power, max_power, max_channel)
    state_size = env.get_state_size()
    print("state_size", state_size)
    action_space_n = env.get_action_space()
    print("action_space_n", action_space_n)

    agent = DQNAgent(state_size, action_space_n)
    # agent.load("./save/weights_old")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = []
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state

            total_reward.append(reward)
            print("episode: {}/{}, step: {}, e: {:.2}, reward: {}"
                      .format(e, EPISODES, time, agent.epsilon, reward))
            
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, env)
        
        agent.update_eps()

        mean_reward = sum(total_reward)/len(total_reward)
        write_file("rewards.txt", mean_reward)

        if e % 10 == 0:
            agent.save("./save/weights_old.{}".format(e))