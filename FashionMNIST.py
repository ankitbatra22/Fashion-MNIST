import torch
from torch import nn, optim
import torch.nn.functional as Fn
import helper
from torchvision import datasets, transforms



# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


#NETWORK ARCHITECTURE (three hidden layers and one output unit)
# ( hidden layer, second hidden )
#self.fc4 is output which has 10 units


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)


    def forward(self, x):
        #FLATTEN
        #First change shape x.shape gives batch size
        x = x.view(x.shape[0], -1)
        #PASS the flattened version of the input tensor through the linear transformations and RELU activation
        x = Fn.relu(self.fc1(x))
        x = Fn.relu(self.fc2(x))
        x = Fn.relu(self.fc3(x))

        #NOW USE  a log softmax with a dimension set to one
        x = Fn.log_softmax(self.fc4(x), dim = 1)

        return x


#create our model
model = Classifier()
#Define our criterion with the negative log likelihood loss which is loss = -log(y)
#If we use log softmax as the output, we use NLLoss as the criterion
Criterion = nn.NLLLoss()
#same as gradient descent but uses momentum to speed up fitting process and adjusts learning rate
optimizer = optim.Adam(model.parameters(), lr = 0.003)


#TRAINING NETWORK

epochs = 300

for e in range(epochs):
    running_loss = 0
    #get the images
    for images, labels in trainloader:
    #Get the log probability by passing in the images in to the model
        logps = model(images)
        loss = Criterion(logps, labels)

    #calculating gradients (FROM MNIST)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print(running_loss)

else:
    print(f"Training loss: {running_loss}") #formatted string for error





dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[1]

ps = torch.exp(model(img))

#helper.view_classify(img, pas, version = "Fasion")






















