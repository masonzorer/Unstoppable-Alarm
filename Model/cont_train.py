# file for continuing training of saved model
import model
import dataloaders
import pandas as pd
import torch

def main():
    model_path = 'model-88.22115384615384.pth' # set chosen model path
    learning_rate = 0.00001
    num_epochs = 25
    batch_size = 16

    # get dataloaders
    train_data = pd.read_csv('Data/train_groundtruth.csv')
    eval_data = pd.read_csv('Data/dev_groundtruth.csv')
    train_dl, dev_dl = dataloaders.create(train_data, eval_data, batch_size)

    model.train(train_dl, dev_dl, num_epochs, learning_rate, model_path)

if __name__ == '__main__':
    main()
