import torch
import torch.nn as nn
import sys
import os

from torchaudio import transforms

# 获取当前脚本所在目录的上一级目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(BASE_DIR)
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, Subset, DataLoader
from utils.utils_reg import *
import numpy as np
import os
from itertools import cycle, zip_longest
from PIL import Image
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)
device_ids = [0,1]
device_1 = f'cuda:{device_ids[0]}'
device_2 = f'cuda:{device_ids[1]}'


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Datasets(Dataset):
    def __init__(self, datasetA, datasetB):

        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, index):
        (xA, lA) = (self.datasetA[index][0], torch.tensor(self.datasetA[index][1]))
        (xB, lB) = (self.datasetB[index][0], torch.tensor(self.datasetB[index][1]))
        return (xA, lA), (xB, lB)

    def __len__(self):

        return len(self.datasetA)

class Unlabeled_Datasets(Dataset):
    def __init__(self, datasetA, datasetB, transform=None):

        self.datasetA = datasetA
        self.datasetB = datasetB
        self.transform = transform

    def __getitem__(self, index):

        xA = self.datasetA[index]
        xB = self.datasetB[index]
        if isinstance(xA, tuple):
            xA = xA[0]
        if isinstance(xB, tuple):
            xB = xB[0]

        if isinstance(xA, str):
            xA = Image.open(xA).convert("RGB")
        if isinstance(xB, str):
            xB = Image.open(xB).convert("RGB")

        if self.transform:
            xA = self.transform(xA)
            xB = self.transform(xB)

        if not isinstance(xA, torch.Tensor):
            xA = transforms.ToTensor()(xA)
        if not isinstance(xB, torch.Tensor):
            xB = transforms.ToTensor()(xB)

        return xA, xB
    def __len__(self):
        return len(self.datasetA)

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
def prepare_data_loaders(
        eo_path, sar_path, batch_size=64, test_size=0.2, num_workers=5, eo_transform= None ,sar_transform = None):

    # Load datasets
    train_data_EO = torchvision.datasets.ImageFolder(root=eo_path, transform=eo_transform)
    train_data_SAR = torchvision.datasets.ImageFolder(root=sar_path, transform=sar_transform)

    # Extract labels and split into labeled and unlabeled sets
    targets = train_data_EO.targets
    indices = np.arange(len(targets))
    labeled_indices, unlabeled_indices = train_test_split(
        indices, test_size=test_size, stratify=targets
    )

    train_data_EO_labeled = Subset(train_data_EO, labeled_indices)
    train_data_EO_unlabeled = Subset(train_data_EO, unlabeled_indices)

    train_data_SAR_labeled = Subset(train_data_SAR, labeled_indices)
    train_data_SAR_unlabeled = Subset(train_data_SAR, unlabeled_indices)

    train_dataset_size_EO = len(train_data_EO)
    train_dataset_size_SAR = len(train_data_SAR)

    # Create weighted random sampler for labeled EO data
    y_train_EO = [train_data_EO_labeled.dataset.targets[i] for i in train_data_EO_labeled.indices]
    assert len(y_train_EO) == len(train_data_EO_labeled), "标签数量与数据数量不一致！"

    class_sample_count = np.array(
        [len(np.where(y_train_EO == t)[0]) for t in np.unique(y_train_EO)]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train_EO])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(
        samples_weight.type(torch.DoubleTensor), len(samples_weight), replacement=True
    )

    # Create datasets
    train_dataset = Datasets(train_data_EO_labeled, train_data_SAR_labeled)
    unlabeled_dataset = Unlabeled_Datasets(train_data_EO_unlabeled, train_data_SAR_unlabeled)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,drop_last=True
    )
    # 把shuffle转成False
    unlabeled_train_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True
    )

    return train_loader, unlabeled_train_loader, train_dataset_size_EO, train_dataset_size_SAR

def train(train_loader, unlabeled_train_loader, device_1, device_2, batch_size=64):
    model_EO = models.efficientnet_b0(pretrained=True)
    num_ftrs_EO = model_EO.classifier[1].in_features
    model_EO.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs_EO, 10))

    for param in model_EO.parameters():
        param.requires_grad = True

    model_SAR = models.efficientnet_b0(pretrained=True)
    num_ftrs_SAR = model_SAR.classifier[1].in_features
    model_SAR.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs_SAR, 10))

    for param in model_SAR.parameters():
        param.requires_grad = True
    alpha_t = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).to(device_1)
    criterion_ce = FocalLoss(alpha_t, 2)
    criterion_da = da_loss()
    alpha = 0.8
    beta = 0.2

    activation_EO = {}

    def getActivation_EO(name):
        def hook(model, input, output):
            activation_EO[name] = output.detach()

        return hook

    activation_SAR = {}

    def getActivation_SAR(name):
        def hook(model, input, output):
            activation_SAR[name] = output.detach()

        return hook

    h1_EO = model_EO.features[8].register_forward_hook(getActivation_EO('8'))
    h1_SAR = model_SAR.features[8].register_forward_hook(getActivation_SAR('8'))

    model_EO.to(device_1)
    model_SAR.to(device_2)

    optim_EO = optim.Adam(model_EO.parameters(), lr=0.003)
    optim_SAR = optim.Adam(model_SAR.parameters(), lr=0.003)
    scheduler_EO = ReduceLROnPlateau(optim_EO, 'min', patience=7)
    scheduler_SAR = ReduceLROnPlateau(optim_SAR, 'min', patience=7)

    print("Starting training on EO and SAR data")
    print("\n🚀 Starting Training...\n")
    for epoch in tqdm(range(30), desc="Epoch Progress", unit="epoch"):  # 添加 epoch 级进度条
        train_loss_EO, train_loss_SAR, correct_EO, correct_SAR, total_EO, total_SAR = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        train_loader_iter = zip_longest(train_loader, unlabeled_train_loader, fillvalue=None)
        progress_bar = tqdm(train_loader_iter, desc=f"Epoch {epoch + 1}", unit="batch", leave=False)  # batch 级进度条

        for (data_labeled, data_unlabeled) in progress_bar:
            if data_labeled is None or data_unlabeled is None:
                continue
            (inputs_EO, labels_EO), (inputs_SAR, labels_SAR) = data_labeled
            inputs_EO, labels_EO = inputs_EO.to(device_1), labels_EO.to(device_1)
            inputs_SAR, labels_SAR = inputs_SAR.to(device_2), labels_SAR.to(device_2)
            inputs_EO_unlabeled, inputs_SAR_unlabeled = data_unlabeled
            inputs_EO_unlabeled = inputs_EO_unlabeled.to(device_1)
            inputs_SAR_unlabeled = inputs_SAR_unlabeled.to(device_2)

            # [64,10]
            outputs_EO = model_EO(inputs_EO)
            outputs_SAR = model_SAR(inputs_SAR).to(device_1)
            h1 = []
            h1.append(activation_EO['8'].to(device_1))
            h1.append(activation_SAR['8'].to(device_1))
            # [64,10]
            outputs_EO_unlabeled = model_EO(inputs_EO_unlabeled)
            outputs_SAR_unlabeled = model_SAR(inputs_SAR_unlabeled)

            h1.append(activation_EO['8'].to(device_1))
            h1.append(activation_SAR['8'].to(device_1))

            loss_ce_EO = criterion_ce(outputs_EO.to(device_1), labels_EO.to(device_1))
            loss_ce_SAR = criterion_ce(outputs_SAR.to(device_1), labels_SAR.to(device_1))

            loss_da = criterion_da(h1[0].to(device_1), h1[1].to(device_1)).to(device_1)
            loss_da_unlabeled = criterion_da(h1[2].to(device_1), h1[3].to(device_1)).to(device_1)

            loss_EO = loss_ce_EO + loss_ce_SAR + ((alpha * loss_da) + (beta * loss_da_unlabeled))
            loss_SAR = loss_ce_SAR + loss_ce_EO + ((alpha * loss_da) + (beta * loss_da_unlabeled))

            optim_EO.zero_grad()
            loss_EO.backward(retain_graph=True, inputs=list(model_EO.parameters()))
            optim_EO.step()
            optim_SAR.zero_grad()
            loss_SAR.backward(inputs=list(model_SAR.parameters()))
            optim_SAR.step()

            predictions_EO = outputs_EO.argmax(dim=1, keepdim=True).squeeze()
            correct_EO += (predictions_EO == labels_EO).sum().item()
            total_EO += labels_EO.size(0)

            predictions_SAR = outputs_SAR.argmax(dim=1, keepdim=True).squeeze()
            predictions_SAR = predictions_SAR.to(device_2)
            labels_SAR = labels_SAR.to(device_2)
            correct_SAR += (predictions_SAR == labels_SAR).sum().item()
            total_SAR += labels_SAR.size(0)

            train_loss_EO += loss_EO.item()
            train_loss_SAR += loss_SAR.item()

            progress_bar.set_postfix(loss_eo=loss_EO.item(), loss_sar=loss_SAR.item())
        accuracy_EO = correct_EO / total_EO
        accuracy_SAR = correct_SAR / total_SAR
        scheduler_EO.step(train_loss_EO)
        scheduler_SAR.step(train_loss_SAR)
        print('Loss_EO after epoch {:} is {:.2f} and accuracy_EO is {:.2f}'.format(epoch,
                                                                                   (train_loss_EO / len(train_loader)),
                                                                                   100.0 * accuracy_EO))
        print('Loss_SAR after epoch {:} is {:.2f} and accuracy_SAR is {:.2f}'.format(epoch, (
                train_loss_SAR / len(train_loader)), 100.0 * accuracy_SAR))
        print()
        h1_EO.remove()
        h1_SAR.remove()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_SAR.state_dict(),
            'model': model_SAR,
            'optimizer_state_dict': optim_SAR.state_dict()
        }, f'/SAR_cross_domain_efficientB0_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_EO.state_dict(),
            'model': model_EO,
            'optimizer_state_dict': optim_EO.state_dict()
        }, f'/EO_cross_domain_efficientB0_epoch_{epoch}.pth')


    print('Finished Simultaneous Training')
    print()
    torch.save(model_EO, '/EO_cross_domain_efficientB0.pth')
    torch.save(model_SAR, '/SAR_cross_domain_efficientB0.pth')
    print()



if __name__ == "__main__":
    EO_file_pth = '/Unicorn_Dataset/EO_Train'
    SAR_file_pth = '/Unicorn_Dataset/SAR_Train'
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loader, unlabeled_train_loader, train_dataset_size_EO, train_dataset_size_SAR = prepare_data_loaders(
        EO_file_pth, SAR_file_pth, batch_size=64, test_size=0.2, num_workers=5, eo_transform=transform,
        sar_transform=transform
    )
    device_1 = "cuda:6"
    device_2 = "cuda:7"
    train(train_loader, unlabeled_train_loader, device_1, device_2, batch_size=64)