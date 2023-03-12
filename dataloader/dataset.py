import os.path
from .preprocess import Preprocesser
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule
import scipy as sp, scipy.io


class EEG(Dataset):
    def __init__(self,
                 data_dir: str,
                 split:str = "train",
                 transform: Callable =None,
                 **kwargs):
        self.data_dir = data_dir
        self.transforms = transform
        self.recoding_meta = np.loadtxt(self.data_dir+f"/{split}.txt",dtype=str)

    def __len__(self):
        return len(self.recoding_meta)

    def _rescale(self,data, gain, offset):
        return (data-offset)/gain

    def _crop(self,data, size=20000):
        duration = data.shape[1]
        if size < duration:
            delta = duration - size
            start= np.random.randint(0,delta)
            return data[:,start:start+size]
        else:
            return data
    def __getitem__(self, idx):
        recording_file, quality_score = self.recoding_meta[idx]
        # load patient meata to get target values like cpc
        eeg = np.asarray(sp.io.loadmat(self.data_dir+"/"+recording_file+".mat")["val"])
        # header = np.genfromtxt("",dtype=str,skip_header=True,usecols=(2,4))
        # gain and offset values for all eeg recording are 32, 0 respectively
        # gain = np.array([float(x.split('/')[0]) for x in header[:,]])
        # offset = header[:,1].astype(float)
        gain = 32000
        offset = 0.0
        eeg = self._rescale(eeg, gain, offset)
        # TODO: test crop augmentation
        # eeg = self._crop()
        return torch.from_numpy(eeg)

class VAEDataset(LightningDataModule):
    def __init__(self,
                 data_path: str,
                 train_batch_size: int = 64,
                 val_batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool =False,
                 **kwargs):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prepare_data()

    def prepare_data(self):
        train_split_file = self.data_dir + "/" +"train.txt"
        val_split_file = self.data_dir + "/" +"val.txt"
        if (not os.path.isfile(train_split_file)) or (not os.path.isfile(val_split_file)):
            print ("starting data processing")
            pre = Preprocesser(self.data_dir)
            pre._train_val_split()
            pre._min_max()
        else:
            print(f"{train_split_file} already exists")
    def setup(self,stage=None):
        print(f" set dataset up for  {stage}")
        if stage=="predict":
            self.all_dataset = EEG(
            self.data_dir ,
            split="all",
            )
            return
        elif stage=="fit":
            self.train_dataset = EEG(
            self.data_dir,
            split="train",
            )
            self.val_dataset = EEG(
            self.data_dir,
            split="val",
            )
        elif stage=="test":
            self.val_dataset = EEG(
            self.data_dir,
            split="val",
            )
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size= self.train_batch_size,
            num_workers= self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def all_dataloader(self):
        return DataLoader(
            self.all_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )