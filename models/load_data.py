import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

def get_dataloader(data_fp, batch_size=500):
    """
    """
    d_set = FetchMotionDataset(data_fp)
    #print("DSET: ", len(d_set))
    
    train_loader = DataLoader(d_set, batch_size=batch_size)
    return train_loader

class FetchMotionDataset(Dataset):
    """
    """

    def __init__(self, data_folder, num_actions=5):
        # This is how the data is setup
        """self.step_dtype = np.dtype([('xt', np.float32, 2048),
                            ('qt', np.float32, 7),
                            ('qdott', np.float32, 7),
                            ('at', np.float32, 7),
                            ('xt_1', np.uint8, 2048),
                            ('qt_1', np.float32, 7),
                            ('qdott_1', np.float32, 7)])"""
        self.num_actions = num_actions

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.data = None
        for data_fp in os.listdir(data_folder):
            if self.data is None:
                self.data = np.load(os.path.join(data_folder, data_fp))['data']
            else:
                self.data = np.concatenate((np.load(os.path.join(data_folder, data_fp))['data'], self.data), axis=1)
            
            self.trajectory_length = self.data.shape[0]*self.num_actions # num inputs per run
            for i in range(num_actions):
                self.trajectory_length -= i

        # move to tensors
        self.xt = torch.tensor(self.data['xt'], device=self.device)
        self.xt_1 = torch.tensor(self.data['xt_1'], device=self.device)
        self.qt = torch.tensor(self.data['qt'], device=self.device)
        self.at_1 = torch.tensor(self.data['at-1'], device=self.device)
        self.at = torch.tensor(self.data['at'], device=self.device)

        print(self.data.shape)

    def __len__(self):
        num_samples, num_runs = self.data.shape
        return num_runs * self.trajectory_length
        # return len(self.data) * len(self.data[0]) #* self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'joint_state': None,
            'next_state': None,
            'last_action': None,
            'true_action': None,
            'goal': None
        }

        # Get current item
        run = item // self.trajectory_length
        index = (item - run * self.trajectory_length) // self.data.shape[0] #index in n samples
        forward_steps = (item - run * self.trajectory_length) % self.num_actions

        # account for the last few of each run that don't have 
        idx_from_end = self.trajectory_length - item + run*self.trajectory_length - 1
        nextras = self.num_actions*(self.num_actions-1)/2

        if idx_from_end < nextras:
            for i in range(1, self.num_actions):
                if idx_from_end <= nextras - i*(i-1)/2 - 1:
                    forward_steps = int(self.num_actions - 1 - i)
                    sample_from_end = -self.num_actions + nextras - i*(i-1)/2 - idx_from_end
                    index = int(self.data.shape[0] + sample_from_end)
        
        # sample = {
        #     'state': torch.tensor(data_point['xt'], device = self.device),
        #     'next_state': torch.tensor(data_point['xt_1'], device = self.device),
        #     'joint_state': torch.tensor(data_point['qt'], device = self.device),
        #     'last_action': torch.tensor(data_point['at-1'], device = self.device),
        #     'true_action': torch.tensor(data_point['at'], device = self.device),
        #     'goal': torch.tensor(data_point['xt_1'], device = self.device)
        # }
        true_action = torch.zeros(self.num_actions, self.at.shape[-1], device=self.device)

        for i in range(forward_steps+1):
            true_action[i,:] = self.at[index+i, run]
        true_action = true_action.reshape(-1)

        sample = {
            'state': self.xt[index, run],
            'next_state': self.xt_1[index+forward_steps, run],
            'joint_state': self.qt[index, run],
            'last_action': self.at_1[index, run],
            'true_action': true_action,
            'goal': self.xt_1[index, run]
        }


        return sample