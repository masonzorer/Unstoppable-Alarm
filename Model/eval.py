# file for evaluating the model
import pandas as pd
import numpy as np
from torchaudio.transforms import MelSpectrogram
from sklearn.metrics import confusion_matrix, classification_report
import torch
import dataloaders
import model

def get_preds_from_model(model_path, dataloader):
    CNN_model = model.CNN()
    CNN_model.load_state_dict(torch.load(model_path))
    CNN_model.eval()

    # get accuracy on test data
    correct = 0
    total = 0
    # save the predictions
    predictions = []

    with torch.no_grad():
        for spectrograms, labels in dataloader:
            outputs = CNN_model(spectrograms)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print eval progress
            print(f'Progress: {total}/{len(dataloader.dataset)}', end='\r')
    print(f'Accuracy: {100 * correct / total}%')

    # convert predictions to a numpy array
    predictions = torch.cat(predictions, dim=0).numpy()

    return predictions

# ensemble of 5 models
def ensemble(test_dataloader):
    # set model paths
    model1 = 'model-88.46-best.pth'
    model2 = 'model-87.019-best.pth'
    model3 = 'model-86.0576923076923.pth'
    model4 = 'model-85.8173076923077.pth'
    model5 = 'model-88.46153846153847.pth'

    # get predictions from each model
    predictions1 = get_preds_from_model(model1, test_dataloader)
    predictions2 = get_preds_from_model(model2, test_dataloader)
    predictions3 = get_preds_from_model(model3, test_dataloader)
    predictions4 = get_preds_from_model(model4, test_dataloader)
    predictions5 = get_preds_from_model(model5, test_dataloader)

    # choose the most common prediction
    predictions = []
    for i in range(len(predictions1)):
        pred = [predictions1[i], predictions2[i], predictions3[i], predictions4[i], predictions5[i]]
        predictions.append(max(set(pred), key=pred.count))

    predictions = np.array(predictions)

    return predictions


def main():
    
    # load test data
    test_data = pd.read_csv('Data/test_groundtruth.csv')
    audio_dir = 'Data/SoundsDataset/-Eval/eval_audio'

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

    # get predictions from ensemble
    predictions = ensemble(test_dataloader)

    # get preds from individual models
    model1 = 'model-88.46-best.pth' # set chosen model path
    #predictions = get_preds_from_model(model1, test_dataloader)

    # get the test labels
    test_labels = test_data['class'].values
    # create confusion matrix with sklearn
    cm = confusion_matrix(test_labels, predictions)
    print(cm, '\n')
    
    # print report
    print('Classification Report')
    print(classification_report(test_labels, predictions))
    

if __name__ == '__main__':
    main()
            


    