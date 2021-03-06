
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HrsrDataset(Dataset):
    def __init__(self, file_names, lr_folder, hr_folder, target_size, resize_lr=True):
        self.file_names = file_names
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.target_size = target_size
        self.resize_lr = resize_lr

    def __getitem__(self, i):
        try:
            lr_image = Image.open(os.path.join(self.lr_folder, self.file_names[i]))
            hr_image = None if self.hr_folder == None else Image.open(
                os.path.join(self.hr_folder, self.file_names[i]))
        except Exception as e:
            print("error loading image: ", self.file_names[i], e)
            exit(1)

        # print(self.file_names[i])
        if self.resize_lr:
            lr_image = lr_image.resize(self.target_size)
        lr_tensor = transforms.ToTensor()(lr_image)
        if hr_image != None:
            hr_tensor = transforms.ToTensor()(hr_image)
            return (lr_tensor, hr_tensor, self.file_names[i])
        else:
            return (lr_tensor, self.file_names[i])

    def __len__(self):
        return len(self.file_names)
