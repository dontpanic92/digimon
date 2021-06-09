from __future__ import absolute_import
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from hrsr_dataset import HrsrDataset
from algo.srcnn import SRCNN
from algo.srresnet import SRResNet

config = {
    "SRCNN": {
        "lr_folder": "../data/270/",
        "hr_folder": "../data/1080/",
        "batch_size": 1,
        "epochs": 5,
        "target_size": (1920, 1080),
        "resize_input": False,
        "resume_from": None,
    },
    "SRResNet": {
        "lr_folder": "../data/540/",
        "hr_folder": "../data/1080/",
        "batch_size": 1,
        "epochs": 5,
        "target_size": (1920, 1080),
        "resize_input": False,
        "resume_from": None,
    }
}


# model = SRCNN()
ModelClass = SRResNet
model_name = ModelClass.__name__

print("model: ", model_name)

LR_FOLDER = config[model_name]["lr_folder"]
# LR_FOLDER = "../data/540/"
# HR_FOLDER = "../data/4k/"
HR_FOLDER = config[model_name]["hr_folder"]
BATCH_SIZE = config[model_name]["batch_size"]
EPOCHS = config[model_name]["epochs"]
target_size = config[model_name]["target_size"]
resize_input = config[model_name]["resize_input"]
resume_from = config[model_name]["resume_from"]


TEMP_FOLDER = "../data/temp/"
MODEL_SAVE_FOLDER = "../models/"
#target_size = (3840, 2160)

##################################################################################


model_folder = os.path.join(MODEL_SAVE_FOLDER,
                            model_name)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Using {} device'.format(device))
# print(model)
# model = model.to(device)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    setup(rank, world_size)

    model = ModelClass().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    train_files, test_files = generate_train_test(0.1)
    train_dataset = HrsrDataset(
        train_files, LR_FOLDER, HR_FOLDER, target_size, resize_input)
    test_dataset = HrsrDataset(
        test_files, LR_FOLDER, HR_FOLDER, target_size, resize_input)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, sampler=test_sampler)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    train(ddp_model, rank, loss_fn, optimizer, train_loader, test_loader, resume_from)

    cleanup()


def train(model, rank, loss_fn, optimizer, train_loader, test_loader, resume_from=None):
    if resume_from != None:
        model_file = os.path.join(
            model_folder, f'{type(model).__name__}_B{BATCH_SIZE}_E{resume_from-1}.ptm')
        model.load_state_dict(torch.load(model_file))
    else:
        resume_from = 0

    for t in range(resume_from, EPOCHS):
        print(f"Epoch {t+1}")
        train_loop(train_loader, rank, model, loss_fn, optimizer, t)
        test_loop(test_loader, rank, model, loss_fn, t)
        if rank == 0:
            torch.save(model.state_dict(), os.path.join(
                model_folder, f'{type(model).__name__}_B{BATCH_SIZE}_E{t}.ptm'))


def train_loop(dataloader: DataLoader, rank, model, loss_fn, optimizer, t):
    size = len(dataloader.sampler)
    for batch, (X, y, file_name) in enumerate(dataloader):
        X = X.to(rank)
        y = y.to(rank)
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


def test_loop(dataloader, rank, model, loss_fn, t):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y, _ in dataloader:
            X = X.to(rank)
            y = y.to(rank)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Accuracy: Avg loss: {test_loss:>8f} \n")
    loss_file = open(os.path.join(
        model_folder, f'{type(model).__name__}_B{BATCH_SIZE}_E{t}_test_loss.txt'), "w")
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


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")
    else:
        run_demo(main, 2)
