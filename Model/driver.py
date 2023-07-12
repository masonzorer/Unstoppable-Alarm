# main driver for model training
import pandas as pd
import matplotlib.pyplot as plt
import dataloaders
import model

def main():

    # Hyperparameters
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.0001
    
    # load data for the training process
    train_data = pd.read_csv('Data/train_groundtruth.csv')
    eval_data = pd.read_csv('Data/dev_groundtruth.csv')

    # create train and eval datasets
    train_dataloader, dev_dataloader = dataloaders.create(train_data, eval_data, batch_size)

    # train the model
    model.train(train_dataloader, dev_dataloader, num_epochs, learning_rate)

if __name__ == '__main__':
    main()