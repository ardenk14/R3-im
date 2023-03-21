import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class GSPNet(nn.Module):
    """
    """

    def __init__(self, state_dim, joint_state_dim, action_dim, lr=1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.joint_state_dim = joint_state_dim
        #self.loss_fcn = nn.MSELoss()

        # Action policy
        self.MLP1 =  nn.Sequential(
          nn.Linear(2 * self.state_dim + self.joint_state_dim + self.action_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, self.action_dim)
        )

        # Forward model
        self.MLP2 =  nn.Sequential(
          nn.Linear(self.state_dim + action_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, self.state_dim)
        )

        params = list(self.MLP1.parameters()) + list(self.MLP2.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def forward(self, state, joint_state, goal, last_action):
        """
        Forward pass through action predictor.
        Action policy network.
        """
        inpt = torch.cat((state, joint_state, goal, last_action), dim=-1)
        pred_next_action = self.MLP1(inpt)
        return pred_next_action
    
    def train_step_full(self, train_loader) -> float:
        """
        Performs an epoch train step for the full model.
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
            state = data['state']
            goal = data['goal']
            joint_state = data['joint_state']
            next_state = data['next_state']
            last_action = data['last_action']
            true_action = data['true_action']

            inpt = torch.cat((state, joint_state, goal, last_action), dim=-1)
            pred_next_action = self.MLP1(inpt)

            inpt2 = torch.cat((state, pred_next_action), dim=-1)
            pred_ns_pred_a = self.MLP2(inpt2)

            inpt3 = torch.cat((state, true_action), dim=-1)
            pred_ns_gt_a = self.MLP2(inpt3)

            # TODO: WHY IS LOSS NEGATIVE? I think true_action must have values between 0 and 1
            loss = F.mse_loss(pred_ns_gt_a, next_state) + F.mse_loss(pred_ns_pred_a, next_state) + F.mse_loss(pred_next_action, true_action) #F.cross_entropy(pred_next_action, true_action)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        return train_loss/len(train_loader)
    
    def train_full_model(self, train_dataloader, num_epochs=100):
        """
        Trains the full model for `num_epochs` epochs.
        You may need to use `train_step` and `val_step`.
        :param train_dataloader: Pytorch DataLoader with the training data.
        :param num_epochs: <int> number of epochs to train the model.
        :param lr: <float> learning rate for the weight update.
        :return:
        """
        pbar = tqdm(range(num_epochs))
        train_losses = []
        for epoch_i in pbar:
            train_loss_i = self.train_step_full(train_dataloader)
            pbar.set_description(f'Train Loss: {train_loss_i:.4f}')
            train_losses.append(train_loss_i)
        return train_losses
    
    def train_step_fwd_only(self, train_loader) -> float:
        """
        Performs an epoch train step for the forward model.
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
            state = data['state']
            #goal = data['goal']
            next_state = data['next_state']
            #last_action = data['last_action']
            true_action = data['true_action']

            inpt = torch.cat((state, true_action), dim=-1)
            pred_ns_gt_a = self.MLP2(inpt)

            loss = F.mse_loss(pred_ns_gt_a, next_state)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        return train_loss/len(train_loader)
    
    def train_forward_only(self, train_dataloader, num_epochs=100):
        """
        Trains the forward model for `num_epochs` epochs.
        :param train_dataloader: Pytorch DataLoader with the training data.
        :param num_epochs: <int> number of epochs to train the model.
        :param lr: <float> learning rate for the weight update.
        :return:
        """
        pbar = tqdm(range(num_epochs))
        train_losses = []
        for epoch_i in pbar:
            train_loss_i = self.train_step_fwd_only(train_dataloader)
            pbar.set_description(f'Train Loss: {train_loss_i:.4f}')
            train_losses.append(train_loss_i)
        return train_losses