# import data reader and neural network from supplied modules
from WindowLoader import DataReader
from Net import Net

# import standard packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# choose hyperparameters
num_epochs = 200 # how many times to train on all samples
batch_size = 32 # how many samples are used for training at a time
train_size = 0.8 # percent of overall data that is for training
learning_rate = 0.0001

# choose which teams to use data from and an amount of time in seconds that each sample or row should span
window_size=0.5
teams=[1,2,3]

# set device, guarantees program will run on GPU if supported
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read files and get size of input vectors according to window size
dataset = DataReader(window_size,teams)
input_size = dataset.X_tens.shape[1]

# get lengths of train and test datasets
train_len = int(len(dataset)*train_size)
test_len = len(dataset) - train_len

# randomly split train and test datasets
train_set, test_set = torch.utils.data.random_split(dataset,[train_len, test_len],
                                                    generator=torch.Generator().manual_seed(42))

# wrap datasets in torch DataLoader to iterate in batches
train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

# Initialize neural network model so that its amount of input nodes corresponds with the window size
model = Net(input_size)

# Set up loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

n_total_steps = len(train_loader)

# training step
model.train()
# train on entire dataset, once for each epoch
for epoch in range(num_epochs):
    for i, (InfraredVecs, Labels) in enumerate(train_loader):
        InfraredVecs = InfraredVecs.to(device)
        Labels = Labels.to(device)

        # before every update zero the gradient
        optimizer.zero_grad()

        # forward pass
        outputs = model(InfraredVecs)

        # compute loss, back propagate loss derivatives through computational graph, and update model weights
        # with optimizer
        loss = criterion(outputs,Labels)
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}")

# evaluate trained model
model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    # first evaluate on train set
    for InfraredVecs, Labels in train_loader:
        InfraredVecs = InfraredVecs.to(device)
        Labels = Labels.to(device)

        # run samples through  model
        outputs = model(InfraredVecs)

        # torch.max returns (value, index). Only the index of the maximum entry in a prediction vector corresponds to
        # predicted response label.
        _, predictions = torch.max(outputs, dim=1)

        # in each batch count number of samples (rows)
        n_samples += Labels.shape[0]
        n_correct += (predictions == Labels).sum().item()

    print(f"Train Set Accuracy = {100 * n_correct / n_samples}")

    # then evaluate on test set
    n_correct = 0
    n_samples = 0
    for InfraredVecs, Labels in test_loader:
        InfraredVecs = InfraredVecs.to(device)
        Labels = Labels.to(device)

        # run test samples through  model
        outputs = model(InfraredVecs)
        _, predictions = torch.max(outputs,dim=1)

        n_samples += Labels.shape[0]
        n_correct += (predictions == Labels).sum().item()

    print(f"Test Set Accuracy = {100 * n_correct / n_samples}")

