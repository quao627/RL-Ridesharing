import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.optimize
from collections import namedtuple
from itertools import count
from environment import *
from newenvironment import *
from gridmap import GridMap
from algorithm import *
from dqn import ReplayMemory, DQN
from dqn import MatchingNetwork
from q_mixer import QMixer
import matplotlib.pyplot as plt
import copy 
from torch.distributions import Categorical
torch.autograd.set_detect_anomaly(True)
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))

class Agent:
    def __init__(self, env, input_size, output_size, hidden_size, mix_hidden = 32, batch_size = 128, lr = 0.001, gamma = .999, eps_start = 0.9, 
                 eps_end = 0.05, eps_decay = 750,  replay_capacity = 10000, num_save = 200, num_episodes = 10000, mode="greedy", training = False, load_file = None):
        self.env = env
        self.orig_env = copy.deepcopy(env)
        self.grid_map = env.grid_map
        self.cars = env.grid_map.cars
        self.num_cars = len(self.cars)
        self.passengers = env.grid_map.passengers
        self.num_passengers = len(self.passengers)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.num_episodes = num_episodes
        self.steps_done = 0
        self.lr = lr
        self.mode = mode
        self.num_save = num_save
        self.training = training
        self.algorithm = NewPairAlgorithm()
        self.episode_durations = []
        self.loss_history = []
        
        self.device = torch.device("cpu")#"cuda:0" if torch.cuda.is_available() else 
        print("Device being used:", self.device)
        # self.policy_net = MatchingNetwork(vehicle_dim, passenger_dim, self.hidden_dim).to(self.device)
        self.policy_net = MatchingNetwork(2, 4, hidden_size).to(self.device)
        self.params = list(self.policy_net.parameters())
        
        if load_file:
            self.policy_net.load_state_dict(torch.load(load_file))
            self.policy_net.eval()
            self.load_file = "Trained_" + load_file
            print("Checkpoint loaded")
        else:         
            self.load_file = self.mode + "_model_num_cars_" + str(self.num_cars) + "_num_passengers_" + str(self.num_passengers) + \
                    "_num_episodes_" + str(self.num_episodes) + "_hidden_size_" + str(self.hidden_size) + ".pth"
            
        self.optimizer = optim.RMSprop(self.params, lr = self.lr)
        #self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1500, gamma=0.1)

        

    def select_action(self, my_car_vec, my_passenger_vec, mode="sample"):
        score_matrix = self.policy_net(my_car_vec.unsqueeze(0), my_passenger_vec.unsqueeze(0)) # num_passengers x num_cars 
        action = [-1] * self.num_passengers
        logprob = torch.zeros(self.num_passengers)
        hisotry = set()
        for i in range(min(self.num_passengers, self.num_cars)):
            mask1 = torch.zeros(self.num_passengers, self.num_cars)
            mask2 = torch.ones(self.num_passengers, self.num_cars)
            for item in hisotry:
                if item >= 0:
                    mask1[item,:] = - torch.ones(self.num_cars) * float("inf")
                    mask2[item,:] = - torch.zeros(self.num_cars)
                else:
                    mask1[:,-item-1] = - torch.ones(self.num_passengers) * float("inf")
                    mask2[:,-item-1] = - torch.zeros(self.num_passengers)
            score_matrix_flattened = score_matrix.view(-1) * mask2.view(-1) + mask1.view(-1)
                        
            prob = F.softmax(score_matrix_flattened,dim=0)
            if mode == "greedy":
                tmp_idx = prob.argmax().item()
            else:
                m = Categorical(prob)
                tmp_idx = m.sample().item()
            pax_idx = tmp_idx // self.num_cars
            car_idx = tmp_idx % self.num_cars
            action[pax_idx] = car_idx
            logprob[pax_idx] = torch.log(prob[tmp_idx])
            hisotry.add(pax_idx)
            hisotry.add(-car_idx-1)
        # loss = logprob.sum()
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # assert 1 == 0
        return action, logprob  

    def random_action(self, state):
        return torch.tensor([[random.randrange(self.num_cars) for car in range(self.num_passengers)]], device=self.device, dtype=torch.long)
    
    
    def get_state(self):
        # Cars (px, py, 1=matched), Passengers(pickup_x, pickup_y, dest_x, dest_y, 1=matched)
        # Vector Size = 3*C + 5*P 
        cars = self.cars
        passengers = self.passengers

        # Encode information about cars
        cars_vec = np.zeros(2*len(cars))
        my_cars_vec = np.zeros((len(cars), 2))
        
        for i, car in enumerate(cars):    
            cars_vec[2*i: 2*i + 2]  = [car.position[0], car.position[1]]
            my_cars_vec[i] = [car.position[0], car.position[1]]

        # Encode information about passengers
        passengers_vec = np.zeros(4*len(passengers))
        my_passengers_vec = np.zeros((len(passengers), 4))
        for i, passenger in enumerate(passengers):
            passengers_vec[4*i: 4*i + 4]  = [passenger.pick_up_point[0], passenger.pick_up_point[1],
                                             passenger.drop_off_point[0],passenger.drop_off_point[1]]
            my_passengers_vec[i] = [passenger.pick_up_point[0], passenger.pick_up_point[1],
                                    passenger.drop_off_point[0],passenger.drop_off_point[1]]

        return torch.tensor(np.concatenate((cars_vec, passengers_vec)), device= self.device, dtype=torch.float).unsqueeze(0), \
            torch.tensor(my_cars_vec, device= self.device, dtype=torch.float).unsqueeze(0), \
            torch.tensor(my_passengers_vec, device= self.device, dtype=torch.float).unsqueeze(0)
    
    
    def train(self):
        self.policy_net.train()
        batch_size = 16
        for episode in range(self.num_episodes):
            actions = []
            logprobs = []
            rewards = []
            for b in range(batch_size):
                self.reset() 
                #self.reset_orig_env()

                state, my_cars_vec, my_passengers_vec = self.get_state()  
                if self.mode == "ours":
                    action, logprob = self.select_action(my_cars_vec[0], my_passengers_vec[0])
                    actions.append(action)
                    logprobs.append(logprob)
                elif self.mode == "greedy":
                    action = [self.algorithm.greedy_fcfs(self.grid_map)]
            
                # action = shape(number of passengers, 1) (-1 if no car is assigned)
                reward = self.env.step([action], self.mode)
                # print(reward)
                rewards.append(reward)
            print("Avg Batch Reward: ", np.array(rewards).sum(axis=1).mean())
            if self.training:
                self.optimize_model(actions, logprobs, rewards)
                # self.plot_durations(self.mode)
                # self.plot_loss_history(self.mode)
             
                
            if self.training and episode % self.num_save == 0:
                torch.save(self.policy_net.state_dict(), "episode_" + str(episode) + "_" +self.load_file )
                print("Checkpoint saved")
                
                    
            print("Episode: ", episode)

           
        if self.training:
            torch.save(self.policy_net.state_dict(), self.load_file )
            print("Checkpoint saved")
        
        print("Finished")  
            
    def reset(self):
        self.env.reset()
        self.grid_map = self.env.grid_map
        self.cars = self.env.grid_map.cars
        self.passengers = self.env.grid_map.passengers
        
    def reset_orig_env(self):

        self.env = copy.deepcopy(self.orig_env)
        self.grid_map = self.env.grid_map
        self.cars = self.env.grid_map.cars
        self.passengers = self.env.grid_map.passengers
        self.grid_map.init_zero_map_cost()

        
    def optimize_model(self, actions, logprobs, rewards):
        loss = torch.zeros(len(actions))
        rewards = np.array(rewards)
        rewards_sum = np.zeros(rewards.shape)
        for step in range(rewards.shape[1]):
            idx = rewards.shape[1] - step - 1
            rewards_sum[:, idx] = rewards[:, idx]
            if idx < rewards.shape[1] - 1:
                rewards_sum[:, idx] += rewards_sum[:, idx+1]
        for batch in range(len(actions)):
            loss[batch] = -(logprobs[batch] * torch.tensor(rewards_sum[batch])).sum()
        loss = loss.mean()
        self.loss_history.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def plot_durations(self, filename):
        print("Saving durations plot ...")
        plt.figure(2)
        plt.clf()

        total_steps = np.array(self.episode_durations)

        N = len(total_steps)
        window_size = 200
        if N < window_size:
            total_steps_smoothed = total_steps
        else:
            total_steps_smoothed = np.zeros(N-window_size)

            for i in range(N-window_size):
                window_steps = total_steps[i:i+window_size]
                total_steps_smoothed[i] = np.average(window_steps)

        plt.title('Episode Duration history')
        plt.xlabel('Episode')
        plt.ylabel('Duration')

        plt.plot(total_steps_smoothed)
        np.save("Duration_"+filename, total_steps_smoothed)
        plt.savefig("Durations_history_" + filename)
        
    def plot_loss_history(self, filename):
        print("Saving loss history ...")
        plt.figure(2)
        plt.clf()
        #loss = torch.tensor(self.loss_history, dtype=torch.float)

        total_loss = np.array(self.loss_history)

        N = len(total_loss)
        window_size = 50
        if N < window_size:
            total_loss_smoothed = total_loss
        else:
            total_loss_smoothed = np.zeros(N-window_size)

            for i in range(N-window_size):
                window_steps = total_loss[i:i+window_size]
                total_loss_smoothed[i] = np.average(window_steps)


        plt.title('Loss history')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.plot(self.loss_history)
        np.save("Loss_"+filename, total_loss_smoothed)
        plt.savefig("Loss_history_" + filename)

if __name__ == '__main__':
    num_cars =20
    num_passengers = 20
    
    grid_map = GridMap(1, (100,100), num_cars, num_passengers)
    cars = grid_map.cars
    passengers = grid_map.passengers
    env = NewEnvironment(grid_map)


    input_size = 2*num_cars + 4*num_passengers # cars (px, py), passengers(pickup_x, pickup_y, dest_x, dest_y)
    output_size = num_cars * num_passengers  # num_cars * (num_passengers + 1)
    hidden_size = 256
    #load_file = "episode_49800_qmix_model_num_cars_10_num_passengers_10_num_episodes_50000_hidden_size_128.pth" # 3218 over 1000 episodes
    #load_file = "episode_41000_dqn_model_num_cars_20_num_passengers_25_num_episodes_100000_hidden_size_256.pth" # 3218 over 1000 episodes, 316.509, 16274
    # greedy 3526, 348.731, 17251
    # random 3386, 337.336, 17092
    load_file = None
    #greedy, random, dqn, qmix
    agent = Agent(env, input_size, output_size, hidden_size, load_file = load_file, lr=0.001, mix_hidden = 64, batch_size=128, eps_decay = 20000, num_episodes=1000, mode = "ours", training = True) # 50,000 episodes for full trains
    agent.train()

    
