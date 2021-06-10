from __future__ import absolute_import
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.tensor import Tensor
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision import transforms
from PIL import Image
from hrsr_dataset import HrsrDataset
from models.srcnn import SRCNN
from models.srresnet import SRResNet
from models.srgan import SRGAN, GeneratorLoss, Discriminator

config = {
    "SRGAN": {
        "lr_folder": "../data/540/",
        "hr_folder": "../data/1080/",
        "batch_size": 1,
        "pretrain_epochs": 2,
        "epochs": 2,
        "target_size": (1920, 1080),
        "resize_input": False,
        "resume_from": None,
        "pretrained_generator": None,
    },
}

GPU_COUNT = 1

ModelClass = SRGAN
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
pretrained_generator = config[model_name]["pretrained_generator"]
pretrain_epoches = config[model_name]["pretrain_epochs"]


TEMP_FOLDER = "../data/temp/"
DATA_FOLDER = "../data/"
MODEL_SAVE_FOLDER = "../weights/"
#target_size = (3840, 2160)

##################################################################################


model_folder = os.path.join(MODEL_SAVE_FOLDER,
                            model_name)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

test_image = Image.open(os.path.join(DATA_FOLDER, "1.jpg"))
test_image_tensor = transforms.ToTensor()(test_image).unsqueeze_(0)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 2160, 3840)
    save_image(img, name)
###############################################################################


def main_gan(rank, world_size):
    setup(rank, world_size)

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

    train_loader = DataLoader(
        train_dataset, BATCH_SIZE, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, BATCH_SIZE,
                             shuffle=False, sampler=test_sampler)

    model = SRResNet().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    pretrain_generator(ddp_model, rank, world_size,
                       train_loader, test_loader)

    train(ddp_model, rank, world_size, train_loader, test_loader)

    cleanup()


def train(generator, rank, world_size, train_loader, test_loader):
    generator_loss_fn = GeneratorLoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0001)

    discriminator = DDP(Discriminator().to(rank), device_ids=[rank])
    discriminator_loss_fn = nn.BCELoss()
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)

    for e in range(0, EPOCHS):
        size = len(train_loader.sampler)
        train_batch(train_loader, rank, generator, discriminator, generator_loss_fn,
                    discriminator_loss_fn, generator_optimizer, discriminator_optimizer)
        test_loss = test_generator(test_loader, rank, generator, nn.MSELoss())

        if rank == 0:
            test_input = test_image_tensor.to(rank)
            with torch.no_grad():
                test_output: Tensor = generator(test_input)
                torchvision.utils.save_image(test_output.detach(
                ), os.path.join(TEMP_FOLDER, f"gan_generator_{e}.png"))

            torch.save({'generator_loss': str(test_loss)},
                       os.path.join(model_folder, f'SRGAN_loss.json'))

    torch.save(generator.module.state_dict(), os.path.join(
        model_folder, f'SRGAN_generator_{e}.pth'))

    torch.save(discriminator.module.state_dict(), os.path.join(
        model_folder, f'SRGAN_discriminator_{e}.pth'))


def train_batch(dataloader, rank, generator, discriminator, generator_loss_fn, discriminator_loss_fn, generator_optimizer, discriminator_optimizer):

    size = len(dataloader.sampler)
    generator.train()
    discriminator.train()
    for batch, (X, y, file_name) in enumerate(dataloader):
        X = X.to(rank)
        print(X)
        y = y.to(rank)
        generator_output = generator(X)

        discriminator.zero_grad()

        discriminator_output = discriminator(generator_output)
        discriminator_ground_truth = discriminator(X)
        discriminator_loss_fake = discriminator_loss_fn(
            discriminator_output, torch.zeros(1))
        discriminator_loss_fake.backward()

        discriminator_loss_truth = discriminator_loss_fn(discriminator_ground_truth, torch.ones(1))
        discriminator_loss_truth.backward()

        discriminator_loss = discriminator_loss_fake + discriminator_loss_truth
        discriminator_optimizer.step()

        generator.zero_grad()
        generator_loss = generator_loss_fn(X, y, discriminator_output, torch.zeros(1))
        generator_loss.backward()
        generator_optimizer.step()

        current = batch * len(X)
        if batch % 10 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def pretrain_generator(model, rank, world_size, train_loader, test_loader):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for e in range(0, pretrain_epoches):
        print(f"Epoch {e}")
        train_loader.sampler.set_epoch(e)
        pretrain_generator_loop(train_loader, rank, model, loss_fn, optimizer)
        test_generator(test_loader, rank, model, loss_fn)

        if rank == 0:
            test_input = test_image_tensor.to(rank)
            print("test_input: ", test_input.size())
            with torch.no_grad():
                test_output: Tensor = model(test_input)
                torchvision.utils.save_image(test_output.detach(
                ), os.path.join(TEMP_FOLDER, f"generator_{e}.png"))

    torch.save(model.module.state_dict(), os.path.join(
        model_folder, f'{type(model).__name__}_pretrain.pth'))


def pretrain_generator_loop(dataloader, rank, model, loss_fn, optimizer):
    size = len(dataloader.sampler)
    model.train()
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\r",)

    print()


def test_generator(dataloader, rank, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y, _ in dataloader:
            X = X.to(rank)
            y = y.to(rank)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= size
    print(f"Test Error: \n Accuracy: Avg loss: {test_loss:>8f} \n")
    return test_loss


def generate_train_test(test_ratio):
    files = os.listdir(LR_FOLDER)[:11]
    
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
    if n_gpus < GPU_COUNT:
        print(f"Requires at least {GPU_COUNT} GPUs to run, but got {n_gpus}.")
    else:
        run_demo(main_gan, GPU_COUNT)
