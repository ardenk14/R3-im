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

    def __init__(self, data_folder):
        # This is how the data is setup
        """self.step_dtype = np.dtype([('xt', np.float32, 2048),
                            ('qt', np.float32, 7),
                            ('qdott', np.float32, 7),
                            ('at', np.float32, 7),
                            ('xt_1', np.uint8, 2048),
                            ('qt_1', np.float32, 7),
                            ('qdott_1', np.float32, 7)])"""
        
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
            self.trajectory_length = len(self.data)

        # move to tensors
        self.xt = torch.tensor(self.data['xt'], device=self.device)
        self.xt_1 = torch.tensor(self.data['xt_1'], device=self.device)
        self.qt = torch.tensor(self.data['qt'], device=self.device)
        self.at_1 = torch.tensor(self.data['at-1'], device=self.device)
        self.at = torch.tensor(self.data['at'], device=self.device)

        print(self.data.shape)

    def __len__(self):
        return len(self.data) * len(self.data[0]) #* self.trajectory_length

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
        trial = item // self.trajectory_length
        index = item % self.trajectory_length
        data_point = self.data[index, trial]

        # sample = {
        #     'state': torch.tensor(data_point['xt'], device = self.device),
        #     'next_state': torch.tensor(data_point['xt_1'], device = self.device),
        #     'joint_state': torch.tensor(data_point['qt'], device = self.device),
        #     'last_action': torch.tensor(data_point['at-1'], device = self.device),
        #     'true_action': torch.tensor(data_point['at'], device = self.device),
        #     'goal': torch.tensor(data_point['xt_1'], device = self.device)
        # }
        sample = {
            'state': self.xt[index, trial],
            'next_state': self.xt_1[index, trial],
            'joint_state': self.qt[index, trial],
            'last_action': self.at_1[index, trial],
            'true_action': self.at[index, trial],
            'goal': self.xt_1[index, trial]
        }


        return sample