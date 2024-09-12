import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO


# Define the Trading Environment
class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: Prices and volume
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.btc_held = 0
        self.total_profit = 0
        return self._next_observation()

    def _next_observation(self):
        return self.data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']].values

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        reward = 0

        if action == 1:  # Buy
            self.btc_held += 1
            reward = -current_price  # Cost of action

        elif action == 2 and self.btc_held > 0:  # Sell
            self.btc_held -= 1
            reward = current_price  # Reward from selling

        self.total_profit += reward

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self._next_observation(), reward, done, {}

    def render(self):
        # Optional: Implement visualization logic
        pass


# Load your data
df = pd.read_csv('d:/BTCUSD.csv')

# Initialize and train the environment
env = TradingEnv(df)

# Initialize PPO agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=20000)

# Save the model
model.save("ppo_trading_model")