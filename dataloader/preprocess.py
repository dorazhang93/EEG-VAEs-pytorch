import sys
import scipy as sp
import numpy as np
from sklearn.model_selection import train_test_split
import os
from dataloader.helper import find_data_folders, load_text_file, get_recording_ids, get_quality_scores
rng = np.random.RandomState(0)

def _check(header_file, recording_file):
    # Load the header file.
    if (not os.path.isfile(header_file)) or (not os.path.isfile(recording_file)):
        print(f"header file {header_file} or recorfing file {recording_file} doesn't exist!")
        return False

    with open(header_file, 'r') as f:
        header = [l.strip() for l in f.readlines() if l.strip()]

    # Parse the header file.
    record_name = None
    num_signals = None
    sampling_frequency = None
    num_samples = None
    signal_files = list()
    gains = list()
    offsets = list()
    channels = list()
    initial_values = list()
    checksums = list()

    for i, l in enumerate(header):
        arrs = [arr.strip() for arr in l.split(' ')]
        # Parse the record line.
        if i==0:
            record_name = arrs[0]
            num_signals = int(arrs[1])
            sampling_frequency = float(arrs[2])
            num_samples = int(arrs[3])
        # Parse the signal specification lines.
        else:
            signal_file = arrs[0]
            gain = float(arrs[2].split('/')[0])
            offset = int(arrs[4])
            initial_value = int(arrs[5])
            checksum = int(arrs[6])
            channel = arrs[8]
            signal_files.append(signal_file)
            gains.append(gain)
            offsets.append(offset)
            initial_values.append(initial_value)
            checksums.append(checksum)
            channels.append(channel)

    # Check that the header file only references one signal file. WFDB format  allows for multiple signal files, but we have not
    # implemented that here for simplicity.
    num_signal_files = len(set(signal_files))
    if num_signal_files!=1:
        print('The header file {}'.format(header_file) \
            + ' references {} signal files; one signal file expected.'.format(num_signal_files))
        return False

    # Load the signal file.
    head, tail = os.path.split(header_file)
    signal_file = os.path.join(head, list(signal_files)[0])
    data = np.asarray(sp.io.loadmat(signal_file)['val'])

    # Check that the dimensions of the signal data in the signal file is consistent with the dimensions for the signal data given
    # in the header file.
    num_channels = len(channels)
    if np.shape(data)!=(num_channels, num_samples):
        print('The header file {}'.format(header_file) \
            + ' is inconsistent with the dimensions of the signal file.')
        return  False
    # Check that the initial value and checksums for the signal data in the signal file are consistent with the initial value and
    # checksums for the signal data given in the header file.
    for i in range(num_channels):
        if data[i, 0]!=initial_values[i]:
            print('The initial value in header file {}'.format(header_file) \
                + ' is inconsistent with the initial value for channel'.format(channels[i]))
            return False
        if np.sum(data[i, :])!=checksums[i]:
            print('The checksum in header file {}'.format(header_file) \
                + ' is inconsistent with the initial value for channel'.format(channels[i]))
            return False
    return True



class Preprocesser:
    def __init__(self,
                 root: str,
                 folder: str,
                 train_val_split: float = 0.2,
                 ):
        self.root = root
        self.folder = folder
        self.patient_folders = find_data_folders(root+"/"+folder)
        self.split = train_val_split
        self.eeg_segments_list = self._find_segments()
        self.num_patients = len(self.patient_folders)
        self._check_eegs()

    def _find_segments(self):
        eegs_file_lists = list()
        for p_folder in self.patient_folders:
            recording_meta_file = os.path.join(self.root,self.folder,p_folder,p_folder+".tsv")
            recording_meta = load_text_file(recording_meta_file)
            for quality_score, recording_id in zip(get_quality_scores(recording_meta), get_recording_ids(recording_meta)):
                if recording_id == "nan":
                    continue
                if quality_score == "nan":
                    print(quality_score, recording_id)
                    continue
                recording_file = os.path.join(p_folder,recording_id)
                eegs_file_lists.append([recording_file,quality_score])
        return eegs_file_lists

    def _check_eegs(self):
        segments_checked = list()
        for eeg_segment in self.eeg_segments_list:
            eeg_segment_file, _ = eeg_segment
            header_file = self.root + "/" + self.folder + "/" + eeg_segment_file +".hea"
            recording_file = self.root + "/" + self.folder + "/" + eeg_segment_file +".mat"
            if _check(header_file, recording_file):
                segments_checked.append(eeg_segment)
        self.eeg_segments_list = segments_checked


    def _train_val_split(self):
        eeg_train, eeg_val = train_test_split(self.eeg_segments_list,test_size=self.split)
        np.savetxt(self.root+"/"+self.folder+"/all.txt", np.array(self.eeg_segments_list),fmt=str)
        np.savetxt(self.root+"/"+self.folder+"/train.txt", np.array(eeg_train),fmt= str)
        np.savetxt(self.root+"/"+self.folder+"/val.txt", np.array(eeg_val),fmt=str)



if __name__ == "__main__":
    data_dir="/home/daqu/Projects/Physionet/physionet.org/files/i-care/1.0"
    data_name="training"
    train_split_file = data_dir + "/" + data_name + "train.txt"
    val_split_file = data_dir + "/" + data_name + "val.txt"
    if (not os.path.isfile(train_split_file)) or (not os.path.isfile(val_split_file)):
        pre = Preprocesser(data_dir, data_name)
        pre._train_val_split()
