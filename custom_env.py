import numpy as np
import gym
import random

class IoTCommunicationEnv(gym.Env):
    def __init__(self, num_user, number_power, max_power, max_channel, W_U=10**6):
        super(IoTCommunicationEnv, self).__init__()
        self.num_user = num_user
        self.number_power = number_power
        self.max_power = max_power
        self.max_channel = max_channel
        self.W_U = W_U
        self.count = 0
        
        # Define action and observation spaces
        # Định nghĩa các subspace cho action_space
        channel_space = gym.spaces.MultiDiscrete([max_channel + 1]*num_user)  # mỗi user chọn 1 channel 
        power_space = gym.spaces.MultiDiscrete([number_power + 1] * num_user)  # mỗi p_i chọn giá trị từ 0 đến max_power

        # action_space tổng cộng sẽ bao gồm subspace của light_on_off_space và light_power_space
        self.action_space = gym.spaces.Tuple([channel_space, power_space])


        # print(self.action_space)

        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Tuple([gym.spaces.Box(low=0, high=1, shape=(1,)) for _ in range(num_user)]),  # Channel gains
            gym.spaces.Discrete(num_user)  # Gamma_i parameter
        ])
        
        self.state = None  # Current state of the environment
    
    def get_action_size(self):
        # Tính action_size
        self.action_size = ((self.max_channel + 1) ** self.num_user)*( (self.number_power + 1)**self.num_user)

        return self.action_size
        
    
    def get_state_size(self):
        return self.num_user + 1

    def reset(self):
        # Initialize the environment
        channel_gains = [np.random.rand() for _ in range(self.num_user)]
        gamma_i =  np.random.choice([4, 6, 8])
        channel_gains.append(gamma_i)
        self.state = channel_gains
        return self.state

    def step(self, action):
        self.count = self.count + 1
        done = False
        if self.count > 400:
            done = True
            self.count = 0

        # Perform a step in the environment
        # num_channels = self.max_channel * self.num_user
        
        # # Extract channel allocation (binary values)
        # channel_allocation = action[:num_channels].reshape(self.num_user, self.max_channel)
        
        # # Extract power control levels
        # power_levels = action[num_channels:]
        
        # Simulate the communication process and calculate reward
        reward = self._calculate_reward(action)
        
        # Simulate state transition
        new_channel_gains = [np.random.normal(0, 1) for _ in range(self.num_user)]
        gamma_i = self.state[-1]
        new_channel_gains.append(gamma_i)
        self.state = new_channel_gains
        
        return self.state, reward, done, {}
    
    def convert_action(self, action_index):
        
        num_channels = action_index%((self.max_channel + 1) ** self.num_user)
        num_powers = (action_index - num_channels)/((self.max_channel + 1) ** self.num_user)

        
        channel_allocation = self.convert_base(num_channels, self.max_channel + 1)
        power_allocation = self.convert_base(num_powers, self.number_power + 1)
        
        return np.array(channel_allocation + power_allocation)

    def convert_base(self, number, base):
        if number == 0:
            return [0]*self.num_user
        
        digits = []
        negative = False
        
        if number < 0:
            negative = True
            number = abs(number)
        
        while number > 0:
            remainder = number % base
            digits.append(int(remainder))
            number //= base
        
        if negative:
            digits.append("-")

        digits = digits[::-1]

        for _ in range(self.num_user - len(digits)):
            digits.append(0)
        return digits

    def _calculate_reward(self, action_index):
        action = self.convert_action(action_index)
        channel_allocation = action[:self.num_user ]
        power_levels = action[self.num_user:]

        print("action_index: ", action_index, "->", action)

        reward = 0

        channel_select_uniq = []
        check_channael = 0
        for i in channel_allocation:
            if i != 0 and i in channel_select_uniq:
                check_channael = check_channael - 1
            elif i != 0 :
                channel_select_uniq.append(i)
        if check_channael < 0:
            reward += check_channael * 2

        
        check_power = 0
        for i in channel_allocation:
            if i != 0 and power_levels[i - 1]==0 :
                check_power = check_power - 1
        if check_power < 0:
            reward += check_power * 1

        if check_channael < 0 or check_power < 0:
            print("reward: ", reward)
            return reward
        

        for k in range(self.num_user):
            channel_index = channel_allocation[k]
            if channel_index == 0:
                continue
            W_U = self.W_U
            channel_gain = self.state[k]
            d_k = 20 # Khoang cach
            interference = self.state[-1]  # Gamma_i parameter
            noise_variance = 10**(-17)  # Adjust this based on your scenario
            
            data_rate = 1.0 * power_levels[int(channel_index - 1)] / self.number_power * self.max_power * channel_gain * d_k**(-2)/ (interference + W_U * noise_variance)
            reward += W_U * np.log2(1 + data_rate)

        return reward

if __name__ == "__main__":

    # Example usage
    num_user = 4
    number_power = 3
    max_power = 0.0316227766
    max_channel = 4

    env = IoTCommunicationEnv(num_user, number_power, max_power, max_channel)

    # Reset the environment
    initial_state = env.reset()
    print("Initial State:", initial_state)

    print("action_size ",env.get_action_size())

    # Take a random action
    # action = env.action_space.sample()
    # print("action :", action)

    action = random.randint(0, env.action_size)
    new_state, reward, done, info = env.step(action)
    print("New State:", new_state)
    print("Reward:", reward)