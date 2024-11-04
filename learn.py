import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class PigGame:
    def __init__(self):
        self.player_points = 0
        self.player_round
        self.bot_points = 0
        self.bot_round = 0
        self.current_player = 1 # 1 for player 2 for bot
    
    def roll():
        return random.randint(1, 6)
    
    def check_win(self):
        if self.bot_points + self.bot_round >= 100:
            return 2
        elif self.player_points + self.player_round >= 100:
            return 1
        else:
            return 0
    
    def make_move(self, hold):
        if hold and self.current_player == 1:
            self.player_points += self.player_round
            self.player_round = 0
        
        elif hold and self.current_player == 2:
            self.bot_points += self.bot_round
            self.bot_round = 0
        
        elif not hold and self.current_player == 1:
            val = self.roll()
            if val == 1:
                self.player_round = 0
                self.current_player = 3 - self.current_player
            else:
                self.player_round += val

        elif not hold and self.current_player == 2:
            val = self.roll()
            if val == 1:
                self.bot_round = 0
                self.current_player = 3 - self.current_player
            else:
                self.bot_round += val
        
        else:
            print("something went wrong")
            return False
        return True
    
    def get_state(self):
        return [self.bot_points, self.bot_round, self.player_points, self.player_round]

class PigGameNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = PigGameNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

class Agent:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Agent's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = AgentNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Agent Net

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        pass

class AgentNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.online = self.__build_cnn(input_dim, output_dim)

        self.target = self.__build_cnn(input_dim, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False
    
    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
    
    def __build_cnn(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(in_channels=input_dim, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )