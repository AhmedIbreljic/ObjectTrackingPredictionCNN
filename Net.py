import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        # Alter amount of neurons in hidden layers until optimal performance is achieved. The amount of layers can also
        # be decreased, but should not be more than 4.

        # The hidden layers are in between each fully connected fc layer, so with the below example there are 3 hidden
        # layers with the sequence of neuron counts 243 -> 193 -> 93 ->3

        # amount of input layer nodes must always correspond to the size of the sample vector
        self.fc1 = nn.Linear(in_features=input_size, out_features=243)

        # each fc layer is fully connected, so the output size of the previous layer corresponds to the input size of
        # the current layer
        self.fc2 = nn.Linear(243, 193)

        self.fc3 = nn.Linear(193, 93)

        self.fc4 = nn.Linear(93,3)

    def forward(self, x):
        # can try Leaky ReLU activation function to solve vanishing gradient problem if necessary
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # softmax activation function is NOT necessary since nn.CrossEntropyLoss() applies softmax on its own
        return x

