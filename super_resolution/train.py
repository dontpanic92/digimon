from __future__ import absolute_import
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from hrsr_dataset import HrsrDataset
from algo.srcnn import SRCNN

LR_FOLDER = "../data/540/"
HR_FOLDER = "../data/4k/"
BATCH_SIZE = 3
EPOCHS = 10
model = SRCNN()
MODEL_SAVE_FOLDER = "../models/"


##################################################################################


model_folder = os.path.join(MODEL_SAVE_FOLDER,
                            type(model).__name__)
os.makedirs(model_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using {} device'.format(device))
model = model.to(device)


def main():
    train_files, test_files = generate_train_test(0.25)
    train_dataset = HrsrDataset(
        train_files, LR_FOLDER, HR_FOLDER, (3840, 2160))
    test_dataset = HrsrDataset(test_files, LR_FOLDER, HR_FOLDER, (3840, 2160))

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)

    torch.save(model.state_dict(), os.path.join(
        model_folder, f'{type(model).__name__}_B{BATCH_SIZE}_E{EPOCHS}.ptm'))


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y, _) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y, _ in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def generate_train_test(test_ratio):
    files = os.listdir(LR_FOLDER)
    random.shuffle(files)

    test_count = int(len(files) * test_ratio)
    test_files = files[:test_count]
    train_files = files[test_count:]
    return (train_files, test_files)


if __name__ == "__main__":
    main()
