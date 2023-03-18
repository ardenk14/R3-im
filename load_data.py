import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

    def __init__(self, data_fp):
        # This is how the data is setup
        """self.step_dtype = np.dtype([('xt', np.float32, 2048),
                            ('qt', np.float32, 7),
                            ('qdott', np.float32, 7),
                            ('at', np.float32, 7),
                            ('xt_1', np.uint8, 2048),
                            ('qt_1', np.float32, 7),
                            ('qdott_1', np.float32, 7)])"""
        
        self.data = np.load(data_fp)['data']
        print("Data: ", self.data[0, 0]['at'])
        self.trajectory_length = len(self.data)

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
            'next_state': None,
            'last_action': None,
            'true_action': None,
            'goal': None
        }

        # Get current item
        trial = item // self.trajectory_length
        index = item % self.trajectory_length
        data_point = self.data[index, trial]

        sample = {
            'state': data_point['xt'],
            'next_state': data_point['xt_1'],
            'last_action': data_point['at-1'],
            'true_action': data_point['at'],
            'goal': data_point['xt_1']
        }

        return sample