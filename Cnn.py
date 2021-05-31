import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm

# flag preprocess data
REBUILD_DATA = True


class DogsVSCats():
    IMG_SIZE = 50
    # images in dataset are diff all shld be normalized to have same size
    # resize to 50,50
    # give directory loc
    # Change the path accordingly
    CATS = "C:/Users/karth/OneDrive/Desktop/CNN_Classification/PetImages/Cat"
    # Change the path accordingly
    DOGS = "C:/Users/karth/OneDrive/Desktop/CNN_Classification/PetImages/Dog"
    # give them labels with class values
    LABELS = {CATS: 0, DOGS: 1}
    # populate with images and labels
    training_data = []
    catcount = 0
    dogcount = 0
    # we need balance so count

    def make_training_data(self):
        # iterating labels
        for label in self.LABELS:
            print(label)

            # tqdm is a progress bar we know where we are
            for f in tqdm(os.listdir(label)):
                try:

                    # f is the file name
                    # we need the full path so
                    path = os.path.join(label, f)
                    # reading in the img and convert to grey scale
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # Color adds dimension color add channel
                    # wat helps identify is patterns
                    # hence turn to gray scale and minimise nn
                    # resize now to (50,50)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # entering one hot vector[0,1]..convert scalar to this and enter
                    # np.eye can conv to vec (no of vec)[hot index]
                    self.training_data.append(
                        [np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    # when we try to load img so dont throwing an error
                    # then might not be of any good or empty
                    # print(str(e))
                    pass
        # shuffling before saving
        np.random.shuffle(self.training_data)
        # saving
        np.save("training_data.npy", self.training_data)

        # print("Cats:",self.catcount) If you want to see the No. of Cat Pictures
       # print("Dogs:",self.dogcount)  If you want to see the No. of Dog Pictures
if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # ip op and kernel size
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        # flat neural network isnt direct
        # we need to hav atleast one nn
        # we dont know what the flatten shape is so we have to give random data
        # to check the flatten shape and then give it
        # BELOW we give random ip with our img dimension
        # check op find the dimen for flat and give in fc
        x = torch.rand(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        # run data to 3 conv layers to check dim
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 512)
        # 2 classes to op is 2
        self.fc2 = nn.Linear(512, 2)
    # forward func for cnn

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # printing the shape coming out from cnn
        # print(x[0].shape)
        # flatten data ,linearize for fc as v dont know shape cmg from cnn
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


net = Net()


optimizer = optim.Adam(net.parameters(), lr=0.001)
# mean square loss
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
# TO change pixel value from 0-255 to 0-1
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

#VAL / TEST

VAL_PCT = 0.1  # valdidation %
val_size = int(len(X)*VAL_PCT)
# we will use this as num to slice with
# print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

BATCH_SIZE = 100
EPOCHS = 1
for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        # print(i,i+BATCH_SIZE)
        # iterates by batches the whiole X len
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy:", round(correct/total, 3))
