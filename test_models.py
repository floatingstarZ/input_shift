import re
from torch.utils import data
from PIL import Image
from torchvision import transforms
import PIL as pil
import numpy as np
import file_save_load as sl
import data_loader
from build_network import *
import torch
import pickle
import os
import sys


image_folder = './catvsdog/test'
models_folder = './models/'
test_results_folder = './models/test_results/'
if not os.path.exists(test_results_folder):
    os.mkdir(test_results_folder)

# 打开记录文档，记录训练过程
# 建立模型文件夹
label_file_path = './data/labels.txt'
model_folder = sl.ModelFloder()
loss_list = []
test_model = model_folder.load_model()

# 载入模型，记录result文件路径
result_path = test_results_folder + 'result_%s'% model_name + '.pkl'
print('save path: %s' % result_path)


TEST_SIZE = 2000
BATCH_SIZE = 50
IMG_SIZE = 224

image_transform = transforms.Compose([
    transforms.Resize([240, 260]),
    transforms.RandomCrop(IMG_SIZE),  # 先四周填充0，在吧图像随机裁剪成IMG_SIZE
    #transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # R,G,B每层的归一化用到的均值和方差
])

# 设置Dataset， 用于loader

test_data = data_loader.DefaultDataset(label_file_path, transform=image_transform)
# 设置loader， batch为100，打乱
test_loader = data.DataLoader(test_data, BATCH_SIZE, shuffle=False)
print('Data load Success')

# 调用GPU
torch.cuda.set_device(0)
import test_if_cuda_ok
test_if_cuda_ok.test_gpu()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('USE cuda')
else:
    print('USE CPU')

result = []

for [iter, data] in enumerate(test_loader):
    with torch.no_grad():
        [name, data] = data
        inputs = data.to(device)
        outputs = test_model(inputs)
        _, predicts = torch.max(outputs, 1)
        for i, predict in enumerate(predicts):
            index_in_name = int(re.findall('\d+', name[i])[0])
            if float(predict) == 0:
                cate = 'Cat'
            else:
                cate = 'Dog'
            result.append([index_in_name, cate])
        print('iter %d DONE' % iter)
        del predicts
        del inputs
        del outputs
# 记录文件
result_file = open(result_path, 'wb+')
pickle.dump(result, result_file)
result_file.close()

