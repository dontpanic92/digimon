from __future__ import absolute_import
import os
import torch
from torch.utils.data import DataLoader
import torchvision
from hrsr_dataset import HrsrDataset
from models.srresnet import SRResNet
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import torch.multiprocessing as mp
INFER_LR_FOLDER = "../data/Infer_Input_540/"
INFER_OUTPUT_FOLDER = "../data/Infer_Output/"
BATCH_SIZE = 1
MODEL_LOAD_FOLDER = "../weights/"
MODEL_NAME = "SRResNet_pretrain_2_1080_16rb.pth"
#MODEL_NAME = "SRGAN_generator_1.pth"

##################################################################################

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    
model_file = os.path.join(MODEL_LOAD_FOLDER,
                          "SRGAN", MODEL_NAME)

os.makedirs(INFER_OUTPUT_FOLDER, exist_ok=True)

def main(rank, world):
    setup(rank, world)
    resnet = SRResNet()
    resnet.load_state_dict(torch.load(model_file))

    model = DDP(resnet).to(rank)
    files = [f for f in os.listdir(INFER_LR_FOLDER) if not f.startswith('.')]
    infer_dataset = HrsrDataset(files, INFER_LR_FOLDER, None, (1920, 1080), resize_lr = False)
    infer_loader = DataLoader(infer_dataset, BATCH_SIZE, shuffle=False)

    model.eval()

    with torch.no_grad():
        for X, file_name in infer_loader:
            X = X.to(rank)
            pred = model(X)

            print(f"Saving {file_name[0]}")
            torchvision.utils.save_image(pred, os.path.join(INFER_OUTPUT_FOLDER, file_name[0]))
    
    cleanup()

if __name__ == "__main__":
    mp.spawn(main,
             args=(1,),
             nprocs=1,
             join=True)

