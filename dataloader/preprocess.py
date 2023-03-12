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
    try:
        data = np.asarray(sp.io.loadmat(signal_file)['val'])
    except:
        print("broken .mat file")
        return False

    # Check that the dimensions of the signal data in the signal file is consistent with the dimensions for the signal data given
    # in the header file.
    num_channels = len(channels)
    num_channels = 18
    num_samples = 30000
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
        if offsets[i]!=0:
            print(f"wrong offset value {offsets[i]} ")
        if gains[i]!=32:
            print(f"wrong gain value {gains[i]}")

    return True



class Preprocesser:
    def __init__(self,
                 root: str,
                 train_val_split: float = 0.2,
                 ):
        self.root = root
        self.patient_folders = find_data_folders(root)
        self.split = train_val_split
        self.eeg_segments_list = self._find_segments()
        print (f"{len(self.patient_folders)} patients finded")
        self._check_eegs()
        print(f"After checking {len(self.patient_folders)} left")
        self.num_patients = len(self.patient_folders)

    def _find_segments(self):
        eegs_file_lists = list()
        for p_folder in self.patient_folders:
            recording_meta_file = os.path.join(self.root,p_folder,p_folder+".tsv")
            try:
                recording_meta = load_text_file(recording_meta_file)
                for quality_score, recording_id in zip(get_quality_scores(recording_meta), get_recording_ids(recording_meta)):
                    if recording_id == "nan":
                        continue
                    if quality_score == "nan":
                        print(quality_score, recording_id)
                        continue
                    recording_file = os.path.join(p_folder,recording_id)
                    eegs_file_lists.append([recording_file,quality_score])
            except:
                print (f"Failed to load tsv file {recording_meta_file}")
                continue
        return eegs_file_lists

    def _check_eegs(self):
        segments_checked = list()
        for eeg_segment in self.eeg_segments_list:
            eeg_segment_file, _ = eeg_segment
            header_file = self.root + "/" + eeg_segment_file +".hea"
            recording_file = self.root + "/" + eeg_segment_file +".mat"
            if _check(header_file, recording_file):
                segments_checked.append(eeg_segment)
        self.eeg_segments_list = segments_checked


    def _train_val_split(self):
        print("spliting train and val")
        eeg_train, eeg_val = train_test_split(self.eeg_segments_list,test_size=self.split)
        np.savetxt(self.root+"/all.txt", np.array(self.eeg_segments_list),fmt="%s")
        np.savetxt(self.root+"/train.txt", np.array(eeg_train),fmt= "%s")
        np.savetxt(self.root+"/val.txt", np.array(eeg_val),fmt="%s")

    def _min_max(self):
        global_min=np.empty(18)
        global_max = np.empty(18)
        for i in  range(len(self.eeg_segments_list)):
            recording_file=self.eeg_segments_list[i][0]
            eeg = np.asarray(sp.io.loadmat(self.root + "/" + recording_file + ".mat")["val"])
            min= eeg.min(axis=1)
            max=eeg.max(axis=1)
            if i ==0:
                global_min = min
                global_max = max
            else:
                global_min = np.concatenate([np.expand_dims(min,axis=0),np.expand_dims(global_min,axis=0)],axis=0).min(axis=0)
                global_max = np.concatenate([np.expand_dims(max,axis=0),np.expand_dims(global_max,axis=0)],axis=0).max(axis=0)
        print("min values for each channel",global_min)
        print("max values for each channel",global_max)
        np.save(self.root+"/min_max.npy",np.concatenate([np.expand_dims(global_min,axis=0),np.expand_dims(global_max,axis=0)],axis=0))



if __name__ == "__main__":
    data_dir="/home/etlar/daqu/Projects/PhysioNet/physionet.org/files/i-care/1.0/training"
    train_split_file = data_dir + "/" + "train.txt"
    val_split_file = data_dir + "/" + "val.txt"
    pre = Preprocesser(data_dir)
    # pre._train_val_split()
    pre._min_max()
