import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    # TODO implement q network
    def __init__(self, inputs, outputs, hidden):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden,outputs)
        #self.bn2 = nn.BatchNorm1d(output)

    def forward(self, x):

        # TODO implement train
        x2 = F.relu(self.fc1(x))
        out = self.fc2(x2)
        return out
    

class DQN(nn.Module):
# TODO implement q network
    def __init__(self, inputs, outputs, hidden):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden,outputs)
        #self.bn2 = nn.BatchNorm1d(output)

    def forward(self, x):
        
        # TODO implement train
        x2 = F.relu(self.fc1(x))
        out = self.fc2(x2)
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        N = query.shape[0]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(N, self.num_heads, self.head_dim)
        K = K.view(N, self.num_heads, self.head_dim)
        V = V.view(N, self.num_heads, self.head_dim)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = F.softmax(energy, dim=-1) # (N, num_heads, num_pax, num_veh)
        
        return attention


class MatchingNetwork(nn.Module):
    def __init__(self, vehicle_dim, passenger_dim, hidden_dim, num_heads):
        super(MatchingNetwork, self).__init__()
        self.vehicle_mlp = nn.Sequential(
            nn.Linear(vehicle_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.passenger_mlp = nn.Sequential(
            nn.Linear(passenger_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.attention = MultiHeadAttention(hidden_dim, num_heads)
    
    def forward(self, vehicles, passengers):
        vehicle_encoding = self.vehicle_mlp(vehicles)
        passenger_encoding = self.passenger_mlp(passengers)
        # score matrix: shape = (num of passengers, num of vehicles)
        score_matrix = self.attention(passenger_encoding, vehicle_encoding, vehicle_encoding)
        return score_matrix