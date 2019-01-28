from torch import nn
from torch import optim
from torchvision import models
ClassNum = 2

# 如果要自己定义一个ResNet，可以直接仿照resnet50的做法，使用ResNet类。

class CaD(nn.Module):
    def __init__(self):
        super(CaD, self).__init__()
        self.conv1 = self.build_basic(
            3, 48, kernel_size=3, stride=1)
        self.conv2 = self.build_basic(
            48, 72, kernel_size=3, stride=2)
        self.conv2_1 = self.build_basic(
            72, 32, kernel_size=1)
        self.conv3 = self.build_basicBN(
            32, 64, kernel_size=3, stride=2)
        self.conv3_1 = self.build_basic(
            64, 64, kernel_size=1, stride=1)
        self.conv3_2 = self.build_basic(
            64, 16, kernel_size=1, stride=1)
        self.avepool = nn.AvgPool2d(kernel_size=7, stride=2)

        self.fc = nn.Sequential(
                    nn.Linear(9216, 2),
                )
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv2_1 = self.conv2_1(conv2)
        conv3 = self.conv3(conv2_1)
        conv3_1 = self.conv3_1(conv3)
        conv3_2 = self.conv3_2(conv3_1)
        avepool = self.avepool(conv3_2)

        feature = avepool.view(avepool.size(0), -1)
        out = self.fc(feature)
        return x

    def build_basic(self, input_channel, output_channel, **args):
        return nn.Sequential(
            nn.Conv2d(input_channel,output_channel,**args),
            nn.ReLU(inplace=True)
        )
    def build_basicBN(self, input_channel, output_channel, **args):
        return nn.Sequential(
            nn.Conv2d(input_channel,output_channel,**args),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )


basic_model = CaD()
# 定义损失函数和优化方式
# #这个损失函数是softmax和交叉熵的结合，输入是一个任意向量x和一个类别c
# 先对这个向量x，取softmax，得到x[c]的概率，之后，取负对数。
optimizer = optim.SGD(basic_model.fc.parameters(), lr=0.0001,weight_decay=5e-4)
# optimizer = optim.SGD(basic_model.classifier.parameters(), lr=0.0001, weight_decay=5e-4)
loss_fun = nn.CrossEntropyLoss(size_average=False)
