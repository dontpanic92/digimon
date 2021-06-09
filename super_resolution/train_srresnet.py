from __future__ import absolute_import
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from hrsr_dataset import HrsrDataset
from algo.srcnn import SRCNN
from algo.srresnet import SRResNet

LR_FOLDER = "../data/270/"
# LR_FOLDER = "../data/540/"
# HR_FOLDER = "../data/4k/"
HR_FOLDER = "../data/1080/"
TEMP_FOLDER = "../data/temp/"
BATCH_SIZE = 1
EPOCHS = 5
model = SRResNet()
MODEL_SAVE_FOLDER = "../models/"
#target_size = (3840, 2160)
target_size = (1920, 1080)

##################################################################################


model_folder = os.path.join(MODEL_SAVE_FOLDER,
                            type(model).__name__)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using {} device'.format(device))
print(model)
model = model.to(device)


def main():
    train_files, test_files = generate_train_test(0.1)
    train_dataset = HrsrDataset(
        train_files, LR_FOLDER, HR_FOLDER, target_size, False)
    test_dataset = HrsrDataset(test_files, LR_FOLDER, HR_FOLDER, target_size, False)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, loss_fn, optimizer, train_loader, test_loader, None)


def train(model, loss_fn, optimizer, train_loader, test_loader, resume_from = None):
    if resume_from != None:
        model_file = os.path.join(
            model_folder, f'{type(model).__name__}_B{BATCH_SIZE}_E{resume_from-1}.ptm')
        model.load_state_dict(torch.load(model_file))
    else:
        resume_from = 0

    for t in range(resume_from, EPOCHS):
        print(f"Epoch {t+1}")
        train_loop(train_loader, model, loss_fn, optimizer, t)
        test_loop(test_loader, model, loss_fn, t)

        torch.save(model.state_dict(), os.path.join(
            model_folder, f'{type(model).__name__}_B{BATCH_SIZE}_E{t}.ptm'))


def train_loop(dataloader, model, loss_fn, optimizer, t):
    size = len(dataloader.dataset)
    for batch, (X, y, file_name) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current = batch * len(X)
        if batch % 10 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # if batch % 1000 == 999:
            # torch.save(model.state_dict(), os.path.join(model_folder, f'{type(model).__name__}_B{BATCH_SIZE}_E{t}_{current}.ptm'))

def test_loop(dataloader, model, loss_fn, t):
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
    print(f"Test Error: \n Accuracy: Avg loss: {test_loss:>8f} \n")
    loss_file = open(os.path.join(model_folder, f'{type(model).__name__}_B{BATCH_SIZE}_E{t}_test_loss.txt'), "w")
    loss_file.write(str(test_loss))
    loss_file.close()


def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 2160, 3840)
    save_image(img, name)


def generate_train_test(test_ratio):
    files = os.listdir(LR_FOLDER)
    random.shuffle(files)

    test_count = int(len(files) * test_ratio)
    test_files = files[:test_count]
    train_files = files[test_count:]
    return (train_files, test_files)


if __name__ == "__main__":
    main()
