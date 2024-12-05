from PIL import Image
import os
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import random
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

n_cpu = os.cpu_count()
global_seed = 0

class NpyDataset(Dataset):
    def __init__(self, image_folder, output_img_folder, transform=None):
        self.image_folder = image_folder
        self.output_img_folder = output_img_folder
        self.transform = transform
        self.images = os.listdir(image_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_folder, self.images[index])
        output_img_path = os.path.join(self.output_img_folder, self.images[index])

        image = np.load(image_path)
        output_img = np.load(output_img_path)

        if self.transform:
            image, output_img = self.transform(image, output_img)
        return image, output_img


def transform_train(image, output_img, size=(224,224)):
    image = Image.fromarray(image)
    output_img = Image.fromarray(output_img)

    image = TF.resize(image, size)
    output_img = TF.resize(output_img, size, interpolation=transforms.InterpolationMode.NEAREST)

    # # random spin
    # if random.random() > 0.5:
    #     angle = random.choice([90, 180, 270])
    #     image = TF.rotate(image, angle)
    #     mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
    #     output_img = TF.rotate(output_img, angle, interpolation=transforms.InterpolationMode.NEAREST)

    # # random horizontal flip
    # if random.random() > 0.5:
    #     image = TF.hflip(image)
    #     mask = TF.hflip(mask)
    #     output_img = TF.hflip(output_img)

    # to tensor
    image = TF.to_tensor(image)
    output_img = TF.to_tensor(output_img)
    
    return image, output_img


def transform_test(image, output_img, size=(224,224)):
    image = Image.fromarray(image)
    output_img = Image.fromarray(output_img)

    image = TF.resize(image, size)
    output_img = TF.resize(output_img, size, interpolation=transforms.InterpolationMode.NEAREST)
    
    image = TF.to_tensor(image)
    output_img = TF.to_tensor(output_img)

    return image, output_img

dist.init_process_group("nccl")
rank = dist.get_rank()

def get_sampler(dataset_):
    sampler = DistributedSampler(dataset_, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=global_seed)
    return sampler
