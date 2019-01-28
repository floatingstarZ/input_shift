from torch import nn
from torch import optim
from torchvision import models
import os
import re
from torch.utils import data
from PIL import Image
from torchvision import transforms
import PIL as pil
import numpy as np
import file_save_load as sl
import data_loader
from build_network import *
# from model_generator import *
import torch
from test_if_cuda_ok import *

label_file_path = './data/labels.txt'
log_file_path = './model/log.txt'
info_file_path = './model/info.info'
# 建立当前训练的模型的文件夹（精确到天）

# 建立模型文件夹
model_folder = sl.ModelFloder()
# 建立model_built_file，保存建立的信息，现在只有分割验证集的顺序需要保存
log = sl.LogFile(log_file_path)
info = sl.InfoFile(info_file_path)

TOTAL_SIZE = 26000
TRAIN_SIZE = 25000
VALIDATE_SIZE = TOTAL_SIZE - TRAIN_SIZE
BATCH_SIZE = 150
IMG_SIZE = 224

image_transform = transforms.Compose([
    transforms.Resize([240, 260]),
    transforms.RandomCrop(IMG_SIZE),  # 先四周填充0，在吧图像随机裁剪成IMG_SIZE
    #transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # R,G,B每层的归一化用到的均值和方差
])
# 分割训练集与验证集，如果已经存在了初始化文件里，则读出来
train_index = []
validata_index = []
if info.data != None:
    [train_index, validata_index] = info.data
else:
    train_index = list(np.random.choice(range(TRAIN_SIZE + VALIDATE_SIZE), TRAIN_SIZE, replace=False))
    validata_index = [x for x in range(TRAIN_SIZE + VALIDATE_SIZE) if x not in train_index]
    info.dump([train_index, validata_index])

# 设置Dataset， 用于loader
train_data = data_loader.DefaultDataset(label_file_path, transform=image_transform, load_index=train_index)
validata_data = data_loader.DefaultDataset(label_file_path,transform=image_transform, load_index=validata_index)
# 设置loader， batch为BATCH_SIZE，打乱
train_loader = data.DataLoader(train_data, BATCH_SIZE, shuffle=True)
validate_loader = data.DataLoader(validata_data, int(VALIDATE_SIZE/2), shuffle=False)
print('Data load Success')

# 打开记录文档，记录训练过程

EPOCH = 100
PRE_EPOCH = model_folder.epoch # 之前的epoch
loss_list = []
premodel = model_folder.load_model()

# 调用GPU
torch.cuda.set_device(1)
test_gpu()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('USE cuda')
else:
    print('USE CPU')

if premodel:
    basic_model = premodel
    print('load premodel')

basic_model = basic_model.to(device)

def cal_acc(basic_model, inputs, labels):
    correct = 0
    predicts = []
    real_output = []
    count = 0
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = basic_model(inputs)
        real_output = outputs
        # predict为每一行最大的值得下标
        _, predicts = torch.max(outputs, 1)
        correct += (predicts == labels).sum()
        acc = float(correct) / float(len(labels))
        print('acc %f' % acc)
        log.write('acc: %f\n' % acc)
        del inputs
        del outputs
        del predicts
        del acc
    return float(correct)


for epoch in range(PRE_EPOCH, EPOCH):
    epoch_loss: float = 0
    log.write('epoch: %d\n' % epoch)
    train_acc = 0
    print('epoch: %d' % epoch)
    for iter, data in enumerate(train_loader):
        [inputs, labels] = data
        # 应该也可以在传播前面几层的时候使用no_grad模式
        # 在后面分类层开启grad，用于更新梯度。
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = basic_model(inputs)

        loss = loss_fun(outputs, labels)
        loss.backward()
        optimizer.step()
        # 计算predictd的概率
        # nn.Softmax(dim = )

        # 记录损失函数的值
        print('iter: %d, loss: %f' % (iter, loss))
        loss_list.append(loss)
        log.write('iter: %d, loss: %f\n' % (iter, loss))
        epoch_loss = float(epoch_loss + loss)
        # train_acc = train_acc + cal_acc(basic_model, inputs, labels)
        del outputs
        del loss
        del inputs

    if epoch % 3 == 0:
        # basic_model.eval()
        # for validate in validate_loader:
        #     [inputs, labels] = validate
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     cal_acc(basic_model, inputs, labels)
        #     del inputs
        #     del labels
        # shufful_validata_train()
        model_folder.save_model(basic_model)






















