import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

class GoalReconizerNet(nn.Module):

    def __init__(self, state_dim, lr=1e-3) -> None:
        super().__init__()

        self.state_dim = state_dim

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = nn.Sequential(
          nn.BatchNorm1d(2*self.state_dim),
          nn.Linear(2*self.state_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 128),
          nn.ReLU(),
          nn.Linear(128, 1),
          nn.Sigmoid()
        )
        self.model.to(self.device)

        params = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def forward(self, x):
        return self.model(x)
    
    
    def train_step(self, train_loader):
        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            self.optimizer.zero_grad()

            input = torch.cat((data['state1'], data['state2']), dim=-1)
            output = self.model(input)
            target = data['close'].view(-1,1).float().to(self.device)
            # print(input.shape, target.shape, output.shape)

            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        return train_loss/len(train_loader)
    
    def train_goal_recognizer(self, train_dataloader, num_epochs=100):
        pbar = tqdm(range(num_epochs))
        train_losses = []
        for epoch_i in pbar:
            train_loss_i = self.train_step(train_dataloader)
            pbar.set_description(f'Train Loss: {train_loss_i:.4f}')
            train_losses.append(train_loss_i)
        return train_losses


