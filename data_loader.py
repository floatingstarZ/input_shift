import os
import re
from torch.utils import data
from PIL import Image
from torchvision import transforms
import PIL as pil

#这是另外一种方法
from torchvision.datasets import ImageFolder

# label_file_path为空字符串的时候就不返回label了。
class DefaultDataset(data.Dataset):
    # load_index 需要加载的样本下标
    # image_folder: 图片文件夹
    # transform: transforms类型的变量，用transforms.Compose定义
    def __init__(self, label_file_path,
                 transform: transforms=None, load_index=None):
        super(data.Dataset, self).__init__()
        self.load_index = load_index
        self.transform = transform
        self.__has_label = True
        # 加载图像，标签
        if label_file_path:
            self.label_loader = LabelLoader(label_file_path, load_index)
            self.image_loader = ImageLoader(label_file_path, load_index)
            if len(self.image_loader) != len(self.label_loader):
                raise Exception('Number of label images does not match')
        else:
            self.image_loader = ImageLoader(image_folder, load_index)
            self.__has_label = False

    def __getitem__(self, index):
        img = self.image_loader[index]
        if self.transform:
            img = img.convert('RGB')
            img = self.transform(img)
        if self.__has_label:
            label = self.label_loader[index]
            # print(self.image_loader.file_names[index], img.shape)
            return [img, label]
        else:
            return [self.image_loader.file_names[index], img]

    def __len__(self):
        return len(self.image_loader)

class BasicLoader(object):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise Exception('%s FILE does not Exist', file_path)
        self.data = None
        self.path = file_path

    def __getitem__(self, index):
        raise NotImplementedError

class LabelLoader(BasicLoader):
    def __init__(self, label_file_path, load_index=None):
        super(LabelLoader, self).__init__(label_file_path)
        f = open(self.path, 'r')
        self.labels: list = []
        for line in f.readlines():
            if line:
                # print(line)
                self.labels.append(int(line.split()[1]))
        if load_index:
            try:
                self.labels = [self.labels[i] for i in load_index]
            except IndexError:
                raise Exception('load_index in LabelLoader out of range')

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def __add__(self, other):
        return


# 根据form_label生成的标签文件的顺序存储文件路径，
class ImageLoader(BasicLoader):
    def __init__(self, label_file_path, load_index=None):
        super(ImageLoader, self).__init__(label_file_path)
        lables = open(self.path, 'r')
        self.file_names: list = []
        for line in lables.readlines():
            if line:
                self.file_names.append(line.split()[0])
        if load_index:
            try:
                self.file_names = [self.file_names[i] for i in load_index]
            except IndexError:
                raise Exception('load_index in LabelLoader out of range')

    def __getitem__(self, index):
        img_name = self.file_names[index]
        img = pil.Image.open(img_name)
        return img

    def __len__(self):
        return len(self.file_names)

# 生成标签文件。生成的格式为：
# 图像完整路径 标签
def form_label_file(image_folder, label_file_path):
    lable_file = open(label_file_path, 'wt+')
    root = image_folder + '/'
    classes = sorted(os.listdir(root))
    label = 0
    for class_folder in classes:
        # label为0到len(classes) - 1
        # 名字为image
        class_folder = root + class_folder + '/'
        if not os.path.isdir(class_folder):
            continue
        images = sorted(os.listdir(class_folder))
        for image in images:
            lable_file.write(class_folder + image + ' ' + str(label))
            lable_file.write('\n')
        label = label + 1
    lable_file.close()


transform_train = transforms.Compose([
    transforms.Resize([240, 260]),
    transforms.RandomCrop(224),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

if __name__ == '__main__':
    image_folder = './data'
    label_file_path = './data/labels.txt'
    form_label_file(image_folder, label_file_path)
    train_data = DefaultDataset(label_file_path, transform_train)
    loader = data.DataLoader(train_data, 100, True)
    a = 1




