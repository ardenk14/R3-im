import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class BehaviorCloningModel(nn.Module):
    """
    """

    def __init__(self, state_dim, action_dim, lr=1e-4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        #self.loss_fcn = nn.MSELoss()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model =  nn.Sequential(
          nn.BatchNorm1d(self.state_dim),
          nn.Linear(self.state_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
        #   nn.Linear(256, 256),
        #   nn.ReLU(),
          nn.Linear(256, self.action_dim)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, state):
        """
        """
        next_action = self.model(state)
        return next_action
    
    def train_step(self, train_loader) -> float:
        """
        Performs an epoch train step.
        :param model: Pytorch nn.Module
        :param train_loader: Pytorch DataLoader
        :param optimizer: Pytorch optimizer
        :return: train_loss <float> representing the average loss among the different mini-batches.
            Loss needs to be MSE loss.
        """
        train_loss = 0. 

        for batch_idx, data in enumerate(train_loader):
            self.optimizer.zero_grad()

            # TODO: extract data correctly
            r3m_state = data['state']
            joint_state = data['joint_state']
            action = data['true_action']

            state = torch.cat((r3m_state, joint_state), dim=-1)

            pred_action = self.model(state)
            loss = F.l1_loss(pred_action, action) #mse_loss(pred_action, action) 
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        return train_loss/len(train_loader)
    
    def train_model(self, train_dataloader, num_epochs=100):
        """
        Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
        You may need to use `train_step` and `val_step`.
        :param train_dataloader: Pytorch DataLoader with the training data.
        :param num_epochs: <int> number of epochs to train the model.
        :param lr: <float> learning rate for the weight update.
        :return:
        """
        pbar = tqdm(range(num_epochs))
        train_losses = []
        for epoch_i in pbar:
            train_loss_i = self.train_step(train_dataloader)
            pbar.set_description(f'Train Loss: {train_loss_i:.4f}')
            train_losses.append(train_loss_i)
        return train_losses