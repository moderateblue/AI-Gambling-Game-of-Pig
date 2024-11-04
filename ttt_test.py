import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class TicTacToe():
    def __init__(self):
        self.current = 1
        self.board = [0, 0, 0,
                      0, 0, 0,
                      0, 0, 0]

    def reset(self):
        self.board = [0] * 9
        self.current = 1

    def move(self, pos):
        if self.is_valid_move(pos):
            self.board[pos] = self.current
            self.current = 3 - self.current
            return True
        return False
    
    def is_valid_move(self, pos):
        return self.board[pos] == 0
	
    def check_winner(self):
        for winner in range(1, 3):
		
            #rows
            for i in range(0, 9, 3):
                if (self.board[i] == winner) and (self.board[i + 1] == winner) and (self.board[i + 2] == winner):
                    return winner

            #columns
            for i in range(3):
                if (self.board[i] == winner) and (self.board[i + 3] == winner) and (self.board[i + 6] == winner):
                    return winner
        
            #top left to bottom right
            if (self.board[0] == winner) and (self.board[4] == winner) and (self.board[8] == winner):
                return winner
            #top right to bottom left
            elif (self.board[2] == winner) and (self.board[4] == winner) and (self.board[6] == winner):
                return winner

        else:
            full = False
            for i in range(9):
                if (self.board[i] != 0):
                    full = True
                else:
                    full = False
                    return 0
            if (full):
                return 3
        

    def get_state(self):
        return self.board
    
    def get_valid_moves(self):
        valid_moves = []

        for i in range(9):
            if self.is_valid_move(i):
                valid_moves.append(i)
        
        return valid_moves


class TicTacToeNN(nn.Module):
    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.fc1 = nn.Linear(9, 11)
        self.fc2 = nn.Linear(11, 11)
        self.fc3 = nn.Linear(11, 9)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = TicTacToeNN()
optimizer = optim.Adam(model.parameters(), lr=0.2)
criterion = nn.MSELoss()

class Agent:
    def __init__(self, model):
        self.model = model
        self.epsilon = 0.2 # Exploration factor

    def select_action(self, state, valid_moves):
        if len(valid_moves) == 0:
            return None # No valid moves left

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_moves) # Random valid action (exploration)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze()
            # Mask invalid moves
            mask = torch.zeros(9, dtype=torch.bool)
            mask[valid_moves] = 1
            q_values[~mask] = -float('inf')
            return torch.argmax(q_values).item()

    def train(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return

        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()
        next_q_values = self.model(next_states_tensor).max(1)[0]
        target_q_values = rewards_tensor + (1 - dones_tensor) * 0.9 * next_q_values

        loss = criterion(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def play_game(agent, game, replay_buffer):
    game.reset()
    state = game.get_state()
    while game.check_winner() == 0:
        valid_moves = game.get_valid_moves()
        action = agent.select_action(state, valid_moves)
        if action is None: # No valid moves left
            break
        if game.move(action):
            next_state = game.get_state()
            reward = 9999999999 if game.check_winner() == game.current else 0
            reward = -9999999999 if game.check_winner() == 3 - game.current else 0
            replay_buffer.append((state, action, reward, next_state, game.check_winner() != 0))
            state = next_state
        else:
            reward = -999999999 # Penalty for invalid move
            replay_buffer.append((state, action, reward, state, False))

    agent.train(replay_buffer)

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(filepath):
    model = TicTacToeNN()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

def train_agent():
    game = TicTacToe()
    model = TicTacToeNN()
    agent = Agent(model)
    replay_buffer = []

    for episode in range(40000):
        play_game(agent, game, replay_buffer)
        if episode % 100 == 0:
            print(f"Episode {episode} completed")

        # Optionally save the model at regular intervals
        # if episode % 10000 == 0:
        #     save_model(model, f"tic_tac_toe_model_{episode}.pth")

    # Save the final model
    save_model(model, "tic_tac_toe_model_final.pth")

if __name__ == "__main__":
    train_agent()