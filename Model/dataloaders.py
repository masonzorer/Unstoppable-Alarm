# create dataloaders that will process the data
import os 
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt

# create dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform, audio, train=True):
        self.data = data
        self.transform = transform
        self.audio = audio
        self.train = train

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        # get the index and ground truth label from the data
        index = self.data.index[index]
        label = self.data['class'][index]

        # get the id from the data
        id = self.data['id'][index]
        id = str(id)

        # use the id to get the audio file from the audio directory
        audio_file = os.path.join(self.audio, id + '.wav')

        # load the audio file
        waveform, sample_rate = torchaudio.load(audio_file, normalize=True)
        # get the spectrogram
        spectrogram = self.transform(waveform)
        # convert the spectrogram to a log scale
        spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)

        # mask the spectrogram
        if self.train:
            spectrogram = self.add_masking(spectrogram)

        # truncate the spectrogram to 5 seconds
        if spectrogram.shape[2] > 800:
            spectrogram = spectrogram[:, :, :800]
        else:
            spectrogram = torch.nn.functional.pad(spectrogram, (0, 800 - spectrogram.shape[2]))
        
        return spectrogram, label
    
    # add time and frequency masking to the waveform
    def add_masking(self, spectrogram):
        # define max time and frequency masking
        max_time_masking = 5
        max_freq_masking = 5

        # get number of time and frequency masks
        num_time_masks = np.random.randint(1, max_time_masking)
        num_freq_masks = np.random.randint(1, max_freq_masking)

        # apply time masking with specaugment library
        for i in range(num_time_masks):
            spectrogram = torchaudio.transforms.TimeMasking(5)(spectrogram)
        for i in range(num_freq_masks):
            spectrogram = torchaudio.transforms.FrequencyMasking(5)(spectrogram)

        return spectrogram

# create dataloaders for training and evaluation
def create(train_data, eval_data, batch_size):
    # define mel spectrogram transformation
    SAMPLE_RATE = 41000
    mel_spectrogram_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=2048,
        hop_length=256,
        n_mels=64
    )
    
    # set the audio directory
    audio_dir = 'Data/SoundsDataset/-Dev/dev_audio'

    # create train and eval datasets
    train_dataset = Dataset(train_data, mel_spectrogram_transform, audio_dir, train=True)
    eval_dataset = Dataset(eval_data, mel_spectrogram_transform, audio_dir, train=False)

    # create train and eval dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, eval_dataloader

def main():
    # load the data
    train_data = pd.read_csv('Data/train_groundtruth.csv')
    eval_data = pd.read_csv('Data/dev_groundtruth.csv')
    # create dataloaders
    train_dataloader, eval_dataloader = create(train_data, eval_data, 16)

    # graph spectrograms with labels
    for spectrogram, label in eval_dataloader:
        # get the spectrogram and label
        spectrogram = spectrogram[0]
        label = label[0]
        # graph the spectrogram
        plt.imshow(spectrogram[0].numpy())
        plt.title(label)
        plt.show()

if __name__ == '__main__':
    main()