from __future__ import absolute_import
import os
import random
from numpy import save
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from hrsr_dataset import HrsrDataset
from algo.srcnn import SRCNN
from algo.srresnet import SRResNet
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import torch.multiprocessing as mp
INFER_LR_FOLDER = "../data/Infer_Input_540/"
INFER_OUTPUT_FOLDER = "../data/Infer_Output/"
BATCH_SIZE = 1
MODEL_LOAD_FOLDER = "../models/"
MODEL_NAME = "DistributedDataParallel_B1_E1.ptm"

##################################################################################

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print("asdf")
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    
model_file = os.path.join(MODEL_LOAD_FOLDER,
                          "SRResNet", MODEL_NAME)

os.makedirs(INFER_OUTPUT_FOLDER, exist_ok=True)

def main(rank, world):
    setup(rank, world)
    model = DDP(SRResNet()).to(rank)
    files = [f for f in os.listdir(INFER_LR_FOLDER) if not f.startswith('.')]
    infer_dataset = HrsrDataset(files, INFER_LR_FOLDER, None, (1920, 1080))
    infer_loader = DataLoader(infer_dataset, BATCH_SIZE, shuffle=False)

    model.load_state_dict(torch.load(model_file))
    model.eval()

    with torch.no_grad():
        for X, file_name in infer_loader:
            X = X.to(rank)
            pred = model(X)
            # im = transforms.ToPILImage()(pred[0].cpu().data)

            print(f"Saving {file_name[0]}")
            # im.save(os.path.join(INFER_OUTPUT_FOLDER, file_name[0]), "PNG")
            save_decoded_image(pred.cpu().data, os.path.join(INFER_OUTPUT_FOLDER, file_name[0]))
    
    cleanup()

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 2160, 3840)
    save_image(img, name)

if __name__ == "__main__":
    mp.spawn(main,
             args=(1,),
             nprocs=1,
             join=True)

