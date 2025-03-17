import re

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, models
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import cv2
device_ids = [7]
device = f'cuda:{device_ids[0]}'

class InfDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.imgs_folder = img_folder
        self.transform = transform
        self.img_paths = []

        img_path = self.imgs_folder + '/'
        img_list = os.listdir(img_path)
        img_list.sort()

        self.img_nums = len(img_list)

        for i in range(self.img_nums):
            img_name = img_path + img_list[i]
            self.img_paths.append(img_name)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        if self.transform:
            img = self.transform(img)
        name = os.path.basename(self.img_paths[idx])  # "Gotcha16664030.png"

        match = re.search(r'\d+', name)
        if match:
            image_id = match.group()
        else:
            raise ValueError(f"无法从文件名 {name} 提取 image_id")
        return (img, image_id)

    def __len__(self):
        return self.img_nums

sar_transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert the tensor to PIL image
    transforms.Resize(224),   # Resize the image to the expected input size (224x224)
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale SAR image to 3-channel
    transforms.ToTensor(),    # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize based on SAR image's distribution
])

inf_transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# dataloader of the dataset
img_folder = '/MAVOC_boilerplate/test'
inf_dataset = InfDataset(img_folder, transform=inf_transform)
inf_dataloader = data.DataLoader(inf_dataset, batch_size=64, shuffle=True)


def test():
    model_SAR_model1 = torch.load('cross_model_2resnet.pth')
    model_SAR_model2 = torch.load('cross_model_1.pth')


    model_SAR_model1.to(device)
    model_SAR_model2.to(device)

    image_id_list = []
    class_id_list = []
    score_list = []
    model_SAR_model1.eval()
    model_SAR_model2.eval()

    # 对模型输出进行 z-score 标准化
    def normalize_output(output):
        mean = output.mean(dim=0, keepdim=True)
        std = output.std(dim=0, keepdim=True)
        return (output - mean) / (std + 1e-5)  # 防止除以零
    with torch.no_grad():
        for batch_idx, (img, name) in tqdm(enumerate(inf_dataloader)):
            img = img.to(device)
            output_unlabeled_SAR_Resnet = normalize_output(model_SAR_model1(img))
            output_unlabeled_SAR_Eff_orignial = normalize_output(model_SAR_model2(img))
            output_unlabeled_SAR = torch.add(0.82 *output_unlabeled_SAR_Resnet,0.18 * output_unlabeled_SAR_Eff_orignial )
            score, pseudo_labeled = torch.max(output_unlabeled_SAR, 1)
            for i in range(len(name)):
                image_id_list.append(int(name[i]))
                class_id_list.append(pseudo_labeled[i].cpu().numpy())
                score_list.append(score[i].cpu().numpy())
    if not (len(image_id_list) == len(class_id_list) == len(score_list)):
        raise ValueError(
            f"❌ not match！ image_id_list={len(image_id_list)}, class_id_list={len(class_id_list)}, score_list={len(score_list)}")
    print("📌 type：")
    print(f"type(image_id_list) = {type(image_id_list)}")
    print(f"type(class_id_list) = {type(class_id_list)}")
    print(f"type(score_list) = {type(score_list)}")
    df = pd.DataFrame({'image_id': image_id_list,
                       'class_id': class_id_list,
                       'score': score_list})

    df.to_csv('results.csv', mode='w', index=False, header=True)


test()