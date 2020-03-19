from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
import time
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        action = random.randrange(self.env.action_space.n)

        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action
            with torch.no_grad():
                action = torch.argmax(self.forward(state))
            # values = self.forward(state)
            # max_value = values.max().item()
            # action = ((values == max_value).nonzero().squeeze(0))[1].item()

        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

        
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    # implement the loss function here

    # print("Start to computer loss")
    # start_time = time.perf_counter()
    # loss = 0
    # for i in range(batch_size):
    #     next_state_max_q = target_model.forward(next_state[i]).max()
    #     predict = reward[i]
    #     if done[i].item() != 1:
    #         predict += gamma * next_state_max_q
    #     curr_action = action[i].item()
    #     target = model.forward(state[i]).squeeze(0)[curr_action]
    #     loss += pow(predict - target, 2)

    curr_q_values = model(state)
    next_q_values = model(next_state)
    next_q_state_values = target_model(next_state)

    curr_q_value = curr_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

    expected_q_value = reward + gamma * next_q_value * (1 - done)
    loss = (curr_q_value - autograd.Variable(expected_q_value.data)).pow(2).mean()


    # end_time = time.perf_counter()
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer
        # print("Sample Start")
        # start_time = time.perf_counter()
        # samples = random.sample(self.buffer, batch_size)
        # state = list(map(lambda x: x[0], samples))
        # action = list(map(lambda x: x[1], samples))
        # reward = list(map(lambda x: x[2], samples))
        # next_state = list(map(lambda x: x[3], samples))
        # done = list(map(lambda x: x[4], samples))
        # end_time = time.perf_counter()
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        state = np.concatenate(state)
        next_state = np.concatenate(next_state)
        # if (end_time - start_time) > 1:
        #     print("Sample End, takes " + str(end_time - start_time) + "seconds\n")

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
