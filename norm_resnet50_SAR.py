#!/usr/bin/env python3
import copy
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
torch.cuda.empty_cache()
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # `models/resnet`
BASE_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # `MAVOC_boilerplate`
sys.path.append(BASE_DIR)
from utils.utils_reg import *
device_ids = [6, 7]
device_1 = f'cuda:{device_ids[0]}'
device_2 = f'cuda:{device_ids[1]}'


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
        """
        根据索引加载数据。
        :param index: 索引
        :return: 经过 transform 处理后的 xA 和 xB
        """
        # 如果 datasetA 和 datasetB 是路径列表，则需要加载图像
        xA = self.datasetA[index]
        xB = self.datasetB[index]

        # 如果是 (image, label) 格式的 tuple，取出 image
        if isinstance(xA, tuple):
            xA = xA[0]
        if isinstance(xB, tuple):
            xB = xB[0]

        # 如果是路径，加载图像
        if isinstance(xA, str):
            xA = Image.open(xA).convert("RGB")
        if isinstance(xB, str):
            xB = Image.open(xB).convert("RGB")

        # 应用 transform
        if self.transform:
            xA = self.transform(xA)
            xB = self.transform(xB)

        # 转换为 Tensor，确保返回类型统一
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

def ood_detection(features, threshold=0.8):
    scores = torch.norm(features, dim=1)  # 计算特征范数
    ood_flags = scores > threshold  # 判断是否为 OOD
    return ood_flags, scores

class MultiModalFusion(nn.Module):
    def __init__(self, input_dim=2048, num_heads=2):
        super(MultiModalFusion, self).__init__()
        assert input_dim % num_heads == 0, "input_dim 必须能被 num_heads 整除"
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # 保持通道数一致
            nn.ReLU()
        )
    def forward(self, feature_EO, feature_SAR):
        batch_size, channels, height, width = feature_EO.size()
        seq_length = height * width  # H × W
        assert seq_length >= self.attention.num_heads, "❌ 注意力头数不能大于序列长度"

        # 64，49，2048
        feature_EO = feature_EO.view(batch_size, channels, -1).permute(0, 2, 1)  # (Batch, Seq, Channels)
        feature_SAR = feature_SAR.view(batch_size, channels, -1).permute(0, 2, 1)  # (Batch, Seq, Channels)
        assert feature_EO.shape[
                   2] == self.attention.embed_dim, f"❌ feature_EO embed_dim 错误: 期望 {self.attention.embed_dim}，但得到 {feature_EO.shape[2]}"

        # 执行多头注意力
        fused_features, _ = self.attention(feature_EO, feature_SAR, feature_SAR)
        fused_features = self.fc(fused_features)

        # 恢复原始形状
        fused_features = fused_features.permute(0, 2, 1).view(batch_size, channels, height, width)

        return fused_features
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, eo_dataset, sar_dataset, eo_transform=None, sar_transform=None):
        self.eo_dataset = eo_dataset
        self.sar_dataset = sar_dataset

    def __len__(self):
        return len(self.eo_dataset)

    def __getitem__(self, idx):
        eo_img, eo_label = self.eo_dataset[idx]
        sar_img, sar_label = self.sar_dataset[idx]
        return (eo_img, eo_label), (sar_img, sar_label)

def prepare_data_loaders(
        eo_path, sar_path, batch_size=16, test_size=0.1, num_workers=5, eo_transform= None ,sar_transform = None):

    # Load datasets
    train_data_EO = torchvision.datasets.ImageFolder(root=eo_path, transform=eo_transform)
    train_data_SAR = torchvision.datasets.ImageFolder(root=sar_path, transform=sar_transform)
    # Create paired dataset
    paired_dataset = PairedDataset(train_data_EO, train_data_SAR)

    # Extract labels and split into labeled and unlabeled sets
    targets = train_data_EO.targets
    indices = np.arange(len(targets))
    labeled_indices, unlabeled_indices = train_test_split( indices, test_size=test_size, stratify=targets)

    paired_dataset_labeled = Subset(paired_dataset, labeled_indices)
    paired_dataset_unlabeled = Subset(paired_dataset, unlabeled_indices)
    # Create DataLoaders for labeled and unlabeled data
    train_loader_unlabeled = DataLoader(paired_dataset_unlabeled, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers)

    y_train_EO = [train_data_EO.targets[i] for i in labeled_indices]
    class_sample_count = np.array([len(np.where(np.array(y_train_EO) == t)[0]) for t in np.unique(y_train_EO)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train_EO])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
    train_loader = data.DataLoader(paired_dataset_labeled, batch_size=32, sampler=sampler, num_workers=5)
    train_dataset_size_EO = len(train_data_EO)
    train_dataset_size_SAR = len(train_data_SAR)

    return train_loader, train_loader_unlabeled , train_dataset_size_EO, train_dataset_size_SAR

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_model = self._init_ema_model()

    def _init_ema_model(self):
        """ 初始化 EMA 模型，确保权重独立且不可训练 """
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    def update(self):
        """ EMA 参数更新 """
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.decay).add_((1 - self.decay) * model_param.data)

            # 额外更新 `buffers`，确保 BN 统计信息同步
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), self.model.buffers()):
                ema_buffer.copy_(model_buffer)

    def get_ema_model(self):
        return self.ema_model

def mmd_loss(x, y, sigma=1.0):
    xx = torch.cdist(x, x, p=2)
    yy = torch.cdist(y, y, p=2)
    xy = torch.cdist(x, y, p=2)
    loss = torch.mean(torch.exp(- xx / (2 * sigma ** 2))) + \
           torch.mean(torch.exp(- yy / (2 * sigma ** 2))) - \
           2 * torch.mean(torch.exp(- xy / (2 * sigma ** 2)))
    return loss

def train(train_loader, unlabeled_train_loader, device_1, device_2,batch_size):
    model_EO = models.resnet101(pretrained=True)
    num_ftrs_EO = model_EO.fc.in_features
    model_EO.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs_EO, 10))
    for param in model_EO.parameters():
        param.requires_grad = True

    model_SAR = models.resnet101(pretrained=True)
    num_ftrs_SAR = model_SAR.fc.in_features
    model_SAR.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs_SAR, 10))
    for param in model_SAR.parameters():
        param.requires_grad = True
    model_EO.to(device_1)
    model_SAR.to(device_2)

    optim_EO = optim.AdamW(model_EO.parameters(), lr=0.00001, betas=(0.9, 0.98), weight_decay=5e-5)
    optim_SAR = optim.AdamW(model_SAR.parameters(), lr=0.00001, betas=(0.9, 0.98), weight_decay=5e-5)
    #optim_fusion = optim.Adam(attention_fusion.parameters(), lr=0.003)
    scheduler_EO = CosineAnnealingLR(optim_EO, T_max=50, eta_min=1e-6)
    scheduler_SAR = CosineAnnealingLR(optim_SAR, T_max=50, eta_min=1e-6)
    alpha_t = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).to(device_1)
    criterion_ce = FocalLoss(alpha_t, gamma=2)
    criterion_da = da_loss()
    alpha = 0.8
    beta = 0.2
    activation_EO = {}
    def getActivation_EO(name):
        def hook(model, input, output):
            if output is not None:
                activation_EO[name] = output.detach()
            else:
                print(f"Warning: Activation for {name} is None.")

        return hook

    activation_SAR = {}
    def getActivation_SAR(name):
        def hook(model, input, output):
            activation_SAR[name] = output.detach()

        return hook
    h1_EO = model_EO.layer4.register_forward_hook(getActivation_EO('layer4'))
    h1_SAR = model_SAR.layer4.register_forward_hook(getActivation_SAR('layer4'))
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    ema_EO = EMA(model_EO, decay=0.999)
    ema_SAR = EMA(model_SAR, decay=0.999)

    print("Starting training on EO and SAR data")
    print("\n🚀 Starting Training...\n")
    for epoch in tqdm(range(30), desc="Epoch Progress", unit="epoch"):
        train_loss_EO ,train_loss_SAR ,correct_EO ,correct_SAR ,total_EO ,total_SAR = 0.0,0.0,0.0,0.0,0.0,0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch", leave=False)

        for data_labeled  in progress_bar:
            if data_labeled is None :
                continue
            (inputs_EO, labels_EO), (inputs_SAR, labels_SAR) = data_labeled
            inputs_EO, labels_EO = inputs_EO.to(device_1), labels_EO.to(device_1)
            inputs_SAR, labels_SAR = inputs_SAR.to(device_2), labels_SAR.to(device_2)

            outputs_EO = model_EO(inputs_EO)
            outputs_SAR = model_SAR(inputs_SAR)
            if torch.isnan(outputs_EO).any() or torch.isnan(outputs_SAR).any():
                 print("⚠️ Warning: outputs contain NaN values!")
                 exit()
            h1 = []
            h1.append(activation_EO['layer4'])
            h1.append(activation_SAR['layer4'])

            loss_ce_EO = criterion_ce(outputs_EO, labels_EO).to(device_1)
            loss_ce_SAR = criterion_ce(outputs_SAR.to(device_1), labels_SAR.to(device_1))
            loss_da = criterion_da(h1[0].to(device_1), h1[1].to(device_1)).to(device_1)

            loss_EO = loss_ce_EO + loss_ce_SAR + loss_da
            loss_SAR = loss_ce_EO + loss_ce_SAR + loss_da
            loss_EO = loss_EO.float()
            loss_SAR = loss_SAR.float()

            if torch.isnan(loss_EO).any():
                print("⚠️ Warning: loss_EO has NaN values!")
                exit()
            for name, param in model_EO.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"⚠️ Warning: Parameter {name} contains NaN or Inf values!")
                    exit()
            optim_EO.zero_grad()
            optim_SAR.zero_grad()

            loss_EO.backward(retain_graph=True, inputs=list(model_EO.parameters()))
            torch.nn.utils.clip_grad_norm_(model_EO.parameters(), max_norm=1.0)
            loss_SAR.backward(inputs=list(model_SAR.parameters()))
            torch.nn.utils.clip_grad_norm_(model_SAR.parameters(), max_norm=1.0)
            optim_EO.step()
            optim_SAR.step()

            ema_EO.update()
            ema_SAR.update()


            predictions_EO = outputs_EO.argmax(dim=1, keepdim=True).squeeze()
            correct_EO += (predictions_EO == labels_EO).sum().item()
            total_EO += labels_EO.size(0)

            predictions_SAR = outputs_SAR.argmax(dim=1, keepdim=True).squeeze()
            correct_SAR += (predictions_SAR == labels_SAR).sum().item()
            total_SAR += labels_SAR.size(0)

            train_loss_EO += loss_EO.item()
            train_loss_SAR += loss_SAR.item()

            progress_bar.set_postfix(loss_eo=loss_EO.item(), loss_sar=loss_SAR.item())
            torch.cuda.empty_cache()
        accuracy_EO = correct_EO / total_EO
        accuracy_SAR = correct_SAR / total_SAR
        scheduler_EO.step(train_loss_EO)
        scheduler_SAR.step(train_loss_SAR)
        print('Loss_EO after epoch {:} is {:.2f} and accuracy_EO is {:.2f}'.format(epoch,
                                                                                   (train_loss_EO / len(train_loader)),
                                                                                   100.0 * accuracy_EO))
        print()
        print('Loss_SAR after epoch {:} is {:.2f} and accuracy_SAR is {:.2f}'.format(epoch, (
                train_loss_SAR / len(train_loader)), 100.0 * accuracy_SAR))
        print()

        if (epoch + 1) % 3 == 0:
            model_EO.eval()
            model_SAR.eval()
            correct_unlabeled = 0
            total_unlabeled = 0
            correct_unlabeled_SAR = 0
            total_unlabeled_SAR = 0
            with torch.no_grad():
                for data_unlabeled in unlabeled_train_loader:
                    (inputs_EO_unlabeled, unlabels_EO), (inputs_SAR_unlabeled, unlabels_SAR) = data_unlabeled
                    inputs_EO_unlabeled, unlabels_EO = inputs_EO_unlabeled.to(device_1), unlabels_EO.to(device_1)
                    inputs_SAR_unlabeled,unlabels_SAR = inputs_SAR_unlabeled.to(device_2), unlabels_SAR.to(device_2)
                    outputs_EO_unlabeled = model_EO(inputs_EO_unlabeled)
                    outputs_SAR_unlabeled = model_SAR(inputs_SAR_unlabeled).to(device_1)
                    outputs_total_unlabeled = torch.add(0.8*outputs_EO_unlabeled,0.2* outputs_SAR_unlabeled)
                    predictions_unlabeled = outputs_total_unlabeled .argmax(dim=1, keepdim=True).squeeze()
                    correct_unlabeled += (predictions_unlabeled ==  unlabels_EO).sum().item()
                    total_unlabeled += unlabels_EO.size(0)
            print('Validation after  epoch {:}  accuracy_EO is {:.2f}%'.format(epoch,100.0 * correct_unlabeled/total_unlabeled))
            TNR_EO = (total_unlabeled - correct_unlabeled) / total_unlabeled

            print()
            print(f'Unlabeled TNR_EO: {100.0 * TNR_EO:.2f}')
            print()
            model_EO.train()
            model_SAR.train()
        h1_EO.remove()
        h1_SAR.remove()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_SAR.state_dict(),
            'model': model_SAR,
            'optimizer_state_dict': optim_SAR.state_dict()
        }, f'/resnet101/SAR_cross_domain_resnet50_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_EO.state_dict(),
            'model': model_EO,
            'optimizer_state_dict': optim_EO.state_dict()
        }, f'/resnet101/EO_cross_domain_resnet101_epoch_{epoch}.pth')
        torch.save(ema_EO.get_ema_model().state_dict(), f'/resnet101/resnet101_EO_ema_epoch_{epoch}.pth')
        torch.save(ema_SAR.get_ema_model().state_dict(), f'/resnet101/resnet101_SAR_ema_epoch_{epoch}.pth')

    print('Finished Simultaneous Training')
    print()
    torch.save(model_EO, '/resnet101/EO_cross_domain_resnet50.pth')
    torch.save(model_SAR, '/resnet101/SAR_cross_domain_resnet50.pth')
    print()


if __name__ == "__main__":
    EO_file_pth = '/Unicorn_Dataset/EO_Train'
    SAR_file_pth = '/Unicorn_Dataset/SAR_Train'

    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_loader, unlabeled_train_loader, train_dataset_size_EO, train_dataset_size_SAR = prepare_data_loaders(
        EO_file_pth, SAR_file_pth, batch_size=32, test_size=0.2, num_workers=3, eo_transform=transform,sar_transform=transform
    )
    device_1 = "cuda:6"
    device_2 = "cuda:7"
    train(train_loader, unlabeled_train_loader, device_1, device_2,batch_size=32)