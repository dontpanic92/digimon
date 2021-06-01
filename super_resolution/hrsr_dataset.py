
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HrsrDataset(Dataset):
    def __init__(self, file_names, lr_folder, hr_folder):
        self.file_names = file_names
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder

    def __getitem__(self, i):
        lr_image = Image.open(os.path.join(self.lr_folder, self.file_names[i]))
        hr_image = Image.open(os.path.join(self.hr_folder, self.file_names[i]))

        lr_image = lr_image.resize(hr_image.size)
        return (transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_image))

    def __len__(self):
        return len(self.file_names)
