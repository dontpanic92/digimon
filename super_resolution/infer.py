from __future__ import absolute_import
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from hrsr_dataset import HrsrDataset
from algo.srcnn import SRCNN

INFER_LR_FOLDER = "../data/Infer_540/"
INFER_OUTPUT_FOLDER = "../data/Infer_Output/"
BATCH_SIZE = 3
model = SRCNN()
MODEL_LOAD_FOLDER = "../models/"
MODEL_NAME = "SRCNN_B3_E10.ptm"

##################################################################################


model_file = os.path.join(MODEL_LOAD_FOLDER,
                          type(model).__name__, MODEL_NAME)

os.makedirs(INFER_OUTPUT_FOLDER, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using {} device'.format(device))
model = model.to(device)


def main():
    files = os.listdir(INFER_LR_FOLDER)
    infer_dataset = HrsrDataset(files, INFER_LR_FOLDER, None, (3840, 2160))
    infer_loader = DataLoader(infer_dataset, BATCH_SIZE, shuffle=False)

    model.load_state_dict(torch.load(model_file))
    model.eval()

    with torch.no_grad():
        for X, y, file_name in infer_loader:
            X = X.to(device)
            pred = model(X)
            im = transforms.ToPILImage()(pred)
            im.save(os.path.join(INFER_OUTPUT_FOLDER, file_name), "PNG")


if __name__ == "__main__":
    main()
