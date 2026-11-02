import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
        
class QTrainer:

    def __init__(self, model, lr, gamma):

        self.lr = lr
        self.gamma = gamma
        self.model = model

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr
        )

        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:

            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

            done = (done,)

        # 1. Predict current Q-values
        pred = self.model(state)

        # 2. Create target without gradient tracking
        target = pred.detach().clone()

        # 3. Calculate target Q-values
        with torch.no_grad():

            for idx in range(len(done)):

                Q_new = reward[idx]

                if not done[idx]:

                    Q_new = reward[idx] + self.gamma * torch.max(
                        self.model(next_state[idx])
                    )

                action_idx = torch.argmax(action[idx]).item()

                target[idx][action_idx] = Q_new

        # 4. Calculate loss
        loss = self.criterion(target, pred)

        # 5. Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        # 6. Update network weights
        self.optimizer.step()