import bisect
import warnings
from torch.utils.data.dataset import Dataset
from torch._utils import _accumulate
import numpy as np
import cv2
import torch
import re
import torchvision


#采用的是平均的长宽360, 360（本来是400）
resize_w = 40
resize_h = 40

class customData(Dataset):
    #从文件夹中截取cut_begin到cut_end作为data
    def __init__(self, cut_begin = 0, cut_end = 10):
        names = []
        with open('E:\\Cat VS Dog\\catvsdog\\submission_example.txt', 'r') as label_file:
            names = [re.split('\t|\n', line)[1] for line in label_file]
        train_path = 'E:\\Cat VS Dog\\catvsdog\\train'
        class_name = ['Cat', 'Dog']
        self.labels = []
        self.data_set = []
        # 使用np.all来比较矩阵是否相等，直接使用img == None会出错
        for index, name in enumerate(names):
            if name in class_name:
                if cut_begin < index <= cut_end:
                    img = cv2.imread(train_path + '\\' + name + '.%d' % index + '.jpg')
                    img_resize = cv2.resize(img, (resize_w, resize_h))
                    m = np.max(cv2.resize(img, (resize_w, resize_h)))
                    img_data = np.float64(img_resize.transpose((2, 0, 1))) / m
                    self.data_set.append(img_data)
                    if name == 'Cat':
                        self.labels.append(0)
                        #print('cat %d' % index)
                    elif name == 'Dog':
                        self.labels.append(1)
                       # print('dog %d' % index)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.data_set[item], self.labels[item]

if __name__ == '__main__':
    import xlwt

    # 创建一个Workbook对象，这就相当于创建了一个Excel文件
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)

    i = 2
    trainset = customData(0, 10)
    testset = customData(10, 20)
    print(len(trainset))
    print(len(testset))
    mean_w = 0
    mean_h = 0
    trainset = trainset.data_set
    for img in trainset:
        print(img.shape)
        mean_w += img.shape[0]
        mean_h += img.shape[1]
    print(mean_w / 2000, '|||', mean_h / 2000)
    trainloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=True,
                                              num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    # class_name = 'cat'
    # result = []
    # # 使用np.all来比较矩阵是否相等，直接使用img == None会出错
    # for i in range(30000):
    #     img = cv2.imread(train_path + '\\' + class_name + '.%d' % (i + 1) + '.jpg')
    #     if np.all(img != None):
    #         cv2.imshow('origin', img)
    #         cv2.waitKey(500)
    #         img_resize = cv2.resize(img, (224, 224))
    #         cv2.imshow('resize', img_resize)
    #         cv2.waitKey(500)
    #
    #         m = np.max(img_resize)
    #         img_data = np.float64(img_resize.transpose((2, 0, 1))) / m
    #         img_list = img_data.reshape(1, 3, 224, 224)
    for [img, label] in trainloader:
        print(label)