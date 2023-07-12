# file for evaluating the model
import pandas as pd
from torchaudio.transforms import MelSpectrogram
from sklearn.metrics import confusion_matrix
import torch
import dataloaders
import model

def main():
    
    # load test data
    test_data = pd.read_csv('Data/dev_groundtruth.csv')
    audio_dir = 'Data/SoundsDataset/-Dev/dev_audio'

    # mel spectrogram transformation
    SAMPLE_RATE = 41000
    mel_spectrogram_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=128
    )

    # create test dataset
    test_dataset = dataloaders.Dataset(test_data, mel_spectrogram_transform, audio_dir, train=False)
    # create test dataloader
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # load the model
    CNN_model = model.CNN()
    CNN_model.load_state_dict(torch.load('2model-97.91666666666667.pth'))
    CNN_model.eval()

    # get accuracy on test data
    correct = 0
    total = 0
    # save the predictions
    predictions = []

    with torch.no_grad():
        for spectrograms, labels in test_dataloader:
            outputs = CNN_model(spectrograms)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            predictions.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print eval progress
            print(f'Progress: {total}/{len(test_dataset)}', end='\r')
    print(f'Accuracy: {100 * correct / total}%')

    # convert predictions to a numpy array
    predictions = torch.cat(predictions, dim=0).numpy()
    # get the test labels
    test_labels = test_data['class'].values
    # create confusion matrix with sklearn
    cm = confusion_matrix(test_labels, predictions)
    print(cm)
    
    

if __name__ == '__main__':
    main()
            


    