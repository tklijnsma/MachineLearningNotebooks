import glob
import os.path as osp
from utils import count_events

class DataContainer(object):
    """docstring for DataContainer"""
    def __init__(self):
        super(DataContainer, self).__init__()
        
        self.datadir = '/home/thomas/acceltraining/rotation_224_v1'
        self.data_size = 224 #image width/height
        self.set_files()


    def set_files(self, fraction=1.):
        self.train_files = glob.glob(osp.join(self.datadir, 'train_file_*.h5'))
        self.test_files = glob.glob(osp.join(self.datadir, 'test_file_*.h5'))
        self.val_files = glob.glob(osp.join(self.datadir, 'val_file_*.h5'))

        if fraction < 1.0:
            print('Using a {0} fraction of the available files'.format(fraction))
            half = lambda l: l[:int(len(l) * fraction)] if len(l) * fraction >= 1. else l
            self.train_files = half(self.train_files)
            self.test_files = half(self.test_files)
            self.val_files = half(self.val_files)

        self.n_train_file = len(self.train_files)
        self.n_test_file = len(self.test_files)
        self.n_val_file = len(self.val_files)
        print('n_train_file = {0}, n_test_file = {1}, n_val_file = {1}'.format(self.n_train_file, self.n_test_file, self.n_val_file))

        self.n_train_events = count_events(self.train_files)
        self.n_test_events = count_events(self.test_files)
        self.n_val_events = count_events(self.val_files)
        print('n_train_events = {0}, n_test_events = {1}, n_val_events = {1}'.format(self.n_train_events, self.n_test_events, self.n_val_events))

data = DataContainer()