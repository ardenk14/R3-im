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
                self.data = np.load(os.path.join(data_folder, data_fp))['data'].reshape(-1,1)
            else:
                self.data = np.concatenate((np.load(os.path.join(data_folder, data_fp))['data'].reshape(-1,1), self.data), axis=1)
            
            self.trajectory_length = self.data.shape[0] # num inputs per run

        # move to tensors
        self.q1 = torch.tensor(self.data['q1'].copy(), device=self.device)
        self.r3m1 = torch.tensor(self.data['r3m1'].copy(), device=self.device)
        self.g1 = torch.tensor(self.data['g1'].copy(), device=self.device)
        self.q2 = torch.tensor(self.data['q2'].copy(), device=self.device)
        self.g2 = torch.tensor(self.data['g2'].copy(), device=self.device)

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
        index = item % self.trajectory_length

        sample = {
            'state': self.r3m1[index, run],
            'joint_state': torch.cat((self.q1[index, run], self.g1[index, run].reshape(-1))),
            'true_action': torch.cat((self.q2[index, run] - self.q1[index,run], self.g2[index, run].reshape(-1)))
        }

        return sample