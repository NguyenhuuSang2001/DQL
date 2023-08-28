# -*- coding: utf-8 -*-
from dqn import *

import os

def get_list_name():
    save_directory = "./save"  # Đường dẫn tới thư mục chứa các tệp weights
    file_names = []  # Danh sách để lưu tên tệp weights

    # Lấy danh sách tên tệp trong thư mục save
    for i in range(1000):  # Thay 10 bằng số lượng tệp weights bạn mong muốn
        file_name = "cartpole-dqn.{}".format(i)  # Định dạng tên tệp
        full_path = os.path.join(save_directory, file_name)
        if os.path.isfile(full_path):
            file_names.append(file_name)

    print(file_names)
    return file_names



if __name__ == "__main__":

    # Example usage
    num_user = 4
    number_power = 3
    max_power = 0.0316227766
    max_channel = 3

    env = IoTCommunicationEnv(num_user, number_power, max_power, max_channel)
    state_size = env.get_state_size()
    print("state_size", state_size)
    action_size = env.get_action_size()
    print("action_size", action_size)

    agent = DQNAgent(state_size, action_size)


    NUM_STATE = 1000
    states = deque(maxlen=NUM_STATE)
    for _ in range (NUM_STATE):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        states.append(state)
    
    weight_files = get_list_name()
    rewards = []
    for file in weight_files:
        agent.load(file)
        reward_ = []
        for state in states:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward_.append(reward)
        rewards.append(sum(reward_)/len(reward_))

