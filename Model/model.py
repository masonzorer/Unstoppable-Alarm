# model for training with a convolutional neural network
import torch
import torch.nn as nn
import torch.nn.functional as F

# define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # batch normalization layer
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # relu activation function
        self.relu = nn.ReLU()
        # dropout layer
        self.dropout1 = nn.Dropout(p=0.5)
        # dropout layer
        self.dropout2 = nn.Dropout(p=0.3)
        # flatten layer
        self.flatten = nn.Flatten()
        # fully connected layer
        self.fc1 = nn.Linear(102400, 256)
        # output layer
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # flatten image input
        x = self.flatten(x)
        # dropout
        x = self.dropout1(x)
        # add 1st hidden layer, with relu activation function
        x = self.relu(self.fc1(x))
        # dropout
        x = self.dropout2(x)
        # output layer
        x = self.fc2(x)
        return x
    
# train the model
def train(train_dataloader, dev_dataloader, num_epochs, learning_rate, model_path=None):
    # create the model
    model = CNN()
    if model_path:
        model.load_state_dict(torch.load(model_path))

    # define the loss function and optimizer (binary classification)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    step_count = 0
    total_loss = 0
    for epoch in range(num_epochs):
        for spectrograms, labels in train_dataloader:
            step_count += 1

            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            output = model(spectrograms)
            # calculate loss
            loss = criterion(output, labels)
            total_loss += loss.item()
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            # print loss and accuracy
            if step_count % 10 == 0:
                avg_loss = total_loss / 10
                print('Epoch: {}/{} \tStep: {} \tLoss: {:.4f}'.format(epoch+1, num_epochs, step_count, avg_loss))
                total_loss = 0

            # evaluate every _ steps
            if step_count % 100 == 0:
                print('evaluating...')
                # compute the accuracy over 250 dev samples
                with torch.no_grad():
                    correct = 0
                    total = 0
                    # train loop
                    for spectrograms, labels in train_dataloader:
                        output = model(spectrograms)
                        _, predicted = torch.max(output.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        if total > 400:
                            break
                    train_accuracy = 100 * correct / total

                    correct = 0
                    total = 0
                    # dev loop
                    for spectrograms, labels in dev_dataloader:
                        output = model(spectrograms)
                        _, predicted = torch.max(output.data, 1)
                        print(predicted)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        if total > 400:
                            break
                    dev_accuracy = 100 * correct / total
                    if dev_accuracy > 85:
                        torch.save(model.state_dict(), f'2model-{dev_accuracy}.pth')

                    # print accuracy metrics
                    print('Train Accuracy: {:.2f}% \tDev Accuracy: {:.2f}%'.format(train_accuracy, dev_accuracy))
            

