import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy


class ConvergeEnv(gym.Env):
    def __init__(self):
        super(ConvergeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(7)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=np.float32([1, 0, 0, 0, 0, 0, 0]),
                                            high=np.float32([96, 1, 1, 1, 1, 1, np.inf]),
                                            shape=(7,))
        self.train_data = pd.read_csv('数据/1458/train.bid.all.hb.csv')
        self.day = 6
        self.budget_param = 2  # 预算参数
        self.split_size = 96  # 拆分粒度
        self.time_step_range = self.train_data['96_time_fraction'].unique()

        self.total_budget = []
        for index, day in enumerate(self.train_data['day'].unique()):
            current_day_budget = np.sum(self.train_data[self.train_data.day.isin([day])].market_price)
            self.total_budget.append(current_day_budget)
        self.total_budget = np.divide(self.total_budget, self.budget_param)

        # 初始base_bid以及avg_pctr；可优化为动态修改
        self.initial_lambda_base_bid = pd.read_csv('数据/base_bid.csv').at[1, '{}'.format(self.budget_param)]
        self.initial_avg_pctr = pd.read_csv('数据/base_bid.csv').at[1, 'avg_pctr']
        self.bid_func_lambda = self.initial_avg_pctr / self.initial_lambda_base_bid  # 出价参数
        # obs,状态信息
        self.time_step = 0
        self.current_budget = self.total_budget[self.day - 6]
        self.Candidate_set = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]

        self.win_pctr = 0

    def reset(self):
        ...

        first_data = self.train_data[
            self.train_data['day'].isin([self.day]) & self.train_data['96_time_fraction'].isin([0])]
        bid = self.bid_func(first_data['pctr'])
        bid = np.where(bid >= 300, 300, bid)
        win_data = first_data[first_data.loc[:, 'market_price'] <= bid]

        state_t = 1 / self.split_size

        state_Bt = (self.current_budget - np.sum(win_data['market_price'])) / self.total_budget[self.day - 6]

        state_ROLt = len(first_data)
        state_BCRt = state_Bt - 1  # 初始化时简化了公式
        state_CPMt = np.mean(win_data['market_price'])
        state_WRt = len(win_data) / len(first_data)
        state_rt = np.sum(win_data['pctr'])
        obs = np.array([state_t, state_Bt, state_ROLt, state_BCRt, state_CPMt, state_WRt, state_rt])

        # obs,状态信息
        self.time_step = 0
        self.current_budget = self.total_budget[self.day - 6]
        self.win_pctr = np.sum(win_data['pctr'])

        # 初始base_bid以及avg_pctr；可优化为动态修改
        self.initial_lambda_base_bid = pd.read_csv('数据/base_bid.csv').at[1, '{}'.format(self.budget_param)]
        self.initial_avg_pctr = pd.read_csv('数据/base_bid.csv').at[1, 'avg_pctr']
        self.bid_func_lambda = self.initial_avg_pctr / self.initial_lambda_base_bid  # 出价参数

        first_step_info = {'time_step': 0,
                'action': 0,
                'current_lambda': self.bid_func_lambda,
                'win_pctr': np.sum(win_data['pctr']),
                'remain_budget': self.current_budget}
        # print(first_step_info)
        return obs

    def bid_func(self, pctr):
        return pctr / self.bid_func_lambda

    def step(self, action):
        ...


        obs, obs_info = self._get_observation(action)
        info = {'time_step': self.time_step,

                'action': self.Candidate_set[action],
                'current_lambda': self.bid_func_lambda,
                'win_pctr': obs_info['win_pctr'],
                'remain_budget': self.current_budget}  # 用于记录训练过程中的环境信息,便于观察训练状态

        reward = obs_info['win_pctr']
        done = self._get_done()
        return obs, reward, done, info

    # 根据需要设计相关辅助函数
    def _get_observation(self, action):
        ...
        # 时段修改
        self.time_step += 1

        # lambda修改

        self.bid_func_lambda = self.bid_func_lambda * (self.Candidate_set[action] + 1)

        # 赢标信息修改
        current_data = self.train_data[
            self.train_data['day'].isin([self.day]) & self.train_data['96_time_fraction'].isin([self.time_step_range[self.time_step]])]
        bid = self.bid_func(current_data['pctr'])
        bid = np.where(bid >= 300, 300, bid)
        win_data = current_data[current_data.loc[:, 'market_price'] <= bid]


        # env信息
        self.current_budget = max(0, self.current_budget - np.sum(win_data['market_price']))


        # obs信息修改
        state_t = self.time_step / self.split_size

        tmp_Bt_1 = self.current_budget / self.total_budget[self.day - 6]
        state_Bt = (self.current_budget - np.sum(win_data['market_price'])) / self.total_budget[self.day - 6]

        state_ROLt = len(current_data)
        state_BCRt = (state_Bt - tmp_Bt_1) / tmp_Bt_1 if tmp_Bt_1>0 else 0 # 初始化时简化了公式
        state_CPMt = np.mean(win_data['market_price'])
        state_WRt = len(win_data) / len(current_data)
        state_rt = np.sum(win_data['pctr'])
        obs = np.array([state_t, state_Bt, state_ROLt, state_BCRt, state_CPMt, state_WRt, state_rt])

        self.win_pctr += state_rt

        info = {'win_pctr': np.sum(win_data['pctr'])}

        return obs, info

    def _get_done(self):
        ...
        done = False
        if self.current_budget == 0:
            done = True

        if self.time_step == len(self.time_step_range) - 1:
            done = True
        return done

    def render(self, mode="human"):
        result = {
            'win_pctr': self.win_pctr,
            'remain_budget': self.current_budget,
        }
        return result

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    env = ConvergeEnv()
    model = DQN(MlpPolicy, env, verbose=1)
    print('start learning')
    model.learn(total_timesteps=10000)
    print('start')

    final_result = []
    for i in range(3):
        obs = env.reset()
        while True:
            action, _state = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            print(info)

            if dones:
                result = env.render()
                final_result.append(result)
                break
    for item in final_result:
        print(item)
