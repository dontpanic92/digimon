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

INFER_LR_FOLDER = "../data/Infer_Input/"
INFER_OUTPUT_FOLDER = "../data/Infer_Output/"
BATCH_SIZE = 1
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
    files = [f for f in os.listdir(INFER_LR_FOLDER) if not f.startswith('.')]
    infer_dataset = HrsrDataset(files, INFER_LR_FOLDER, None, (3840, 2160))
    infer_loader = DataLoader(infer_dataset, BATCH_SIZE, shuffle=False)

    model.load_state_dict(torch.load(model_file))
    model.eval()

    with torch.no_grad():
        for X, file_name in infer_loader:
            X = X.to(device)
            pred = model(X)
            # im = transforms.ToPILImage()(pred[0].cpu().data)

            print(f"Saving {file_name[0]}")
            # im.save(os.path.join(INFER_OUTPUT_FOLDER, file_name[0]), "PNG")
            save_decoded_image(pred.cpu().data, os.path.join(INFER_OUTPUT_FOLDER, file_name[0]))

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 3840, 2160)
    save_image(img, name)

if __name__ == "__main__":
    main()
