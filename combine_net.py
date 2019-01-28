from torch import nn
from torch import optim
from torchvision import models
import numpy as np
import torch
import os
ClassNum = 2
# 投票产生最终结果，形成一个最终的模型
class model_vote:
    def __init__(self, model_folder, device):
        self.model_folder: str = model_folder  # model所在文件夹
        self.device = device
        if not os.path.exists(model_folder):
            if not os.path.isdir(model_folder):
                raise Exception('WRONG model folder')
        models = sorted(os.listdir(model_folder))
        models = [os.path.join(model_folder,
                               model) for model in models]
        # 存储模型
        self.models = models

    # 反向传播，得到输出
    def forward(self, input):
        outputs = []
        with torch.no_grad():
            for model in self.models:
                model = torch.load(model).to(self.device)
                outputs.append(model(input))
                del model
        return outputs

    # 直接预测，只适用于猫狗大战
    def predict(self, input):
        with torch.no_grad():
            batch_size = input.size()[0]
            model_num = len(self.models)
            result = np.zeros(batch_size)
            count = 0
            for model in self.models:
                print('model %s' % self.models[count])
                print('begin allocated', torch.cuda.memory_allocated())
                print('begin cached', torch.cuda.memory_cached())
                count = count + 1

                inputs = input.to(self.device)
                print('after input allocated', torch.cuda.memory_allocated())
                print('after input cached', torch.cuda.memory_cached())
                model = torch.load(model).to(self.device)
                print('after load model allocated', torch.cuda.memory_allocated())
                print('after load model cached', torch.cuda.memory_cached())
                output = model(inputs)
                print('after output allocated', torch.cuda.memory_allocated())
                print('after output cached', torch.cuda.memory_cached())
                values, predicts = torch.max(output, 1)
                # result = result + predicts.float().cpu().numpy()
                del model
                del output
                del predicts
                del inputs
                del values
                torch.cuda.empty_cache()
            # print(result)
        # return result
            for i in range(len(result)):
                if result[i] > model_num / 2:
                    result[i] = 1
                else:
                    result[i] = 0
        return result

if __name__ == '__main__':
    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.Resize([240, 260]),
        transforms.RandomCrop(224),  # 先四周填充0，在吧图像随机裁剪成IMG_SIZE
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv = model_vote('./model/2018-12-20', device)
    image_folder = './catvsdog/test'
    import data_loader
    test_data = data_loader.DefaultDataset('', image_folder, transform=image_transform,
                                           load_index=range(20))
    from torch.utils import data
    test_loader = data.DataLoader(test_data, 10, shuffle=False)
    for data in test_loader:
        [name, data] = data
        input = data.to(device)
        out = mv.predict(input)
        print(out)
