import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

parser = argparse.ArgumentParser("説明文")
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

iris = load_iris(as_frame=True)["frame"]
train, test = train_test_split(iris, test_size=30)

X_train = train.iloc[:,:4].to_numpy()
y_train = train.iloc[:,-1].to_numpy()
X_test = test.iloc[:,:4].to_numpy()
y_test = test.iloc[:,-1].to_numpy()

# hyper-params
lr = 0.001
batch_size = 32
max_epochs = 500

net = nn.Sequential(
    nn.Linear(4,20),
    nn.ReLU(),
    nn.Linear(20,3),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

def get_batch(X,Y,batch_size=16, shuffle=True):
    # GPUに対応するために半精度にキャスト
    X = X.astype(np.float32)
    # Xとyをシャッフルする
    index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(index)
        X = X[index]
        Y = Y[index]
    # batch_sizeずつ切り出してタプルとして返すループ
    for i in range(0, y_train.shape[0],batch_size):
        x = X[i:i+batch_size]
        y = Y[i:i+batch_size]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        yield (x,y)

if __name__ == "__main__":
    lr = args.learning_rate
    batch_size = args.batch_size
    seed = args.seed

    print(lr, batch_size, seed)

tmp = []

for epoch in range(max_epochs):
    for batch in get_batch(X_train,y_train):
        optimizer.zero_grad()
        x,y = batch

        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        tmp.append(float(loss))



net = nn.Sequential(
    nn.Linear(4,20),
    nn.ReLU(),
    nn.Linear(20,3),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

net.cuda()

X_val, y_val = X_test, y_test
training_loss = []
valdation_loss = []
for epoch in range(max_epochs):
    net.train() #
    for batch in get_batch(X_train,y_train):
        optimizer.zero_grad()
        x,y = batch
        x = x.to("cuda:0")
        y = y.to("cuda:0")
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        training_loss.append(float(loss))

    net.eval() #
    with torch.no_grad():
        for batch in get_batch(X_val, y_val):
            x,y = batch
            x = x.to("cuda:0")
            y = y.to("cuda:0")
            output = net(x)
            loss = criterion(output, y)
            valdation_loss.append(float(loss))

