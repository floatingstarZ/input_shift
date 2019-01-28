from torchvision import transforms
import torch.nn.functional as F
import torch.autograd as autograd #自动求导
import file_save_load as sl
import torch
from socket_utils import *
import cv2
from PIL import Image
import PIL as pil
import numpy as np
import math

# 加载模型
model_folder = sl.ModelFloder()
model_path = './/test_model//model_20Class.model'
test_model = model_folder.load_model(model_path=model_path)
test_model.eval()
print('load model from : %s' % model_folder.model_path)

# 图像变换
IMG_SIZE = 224
image_transform = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),
    # transforms.RandomCrop(IMG_SIZE),  # 先四周填充0，在吧图像随机裁剪成IMG_SIZE
    #transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # R,G,B每层的归一化用到的均值和方差
])


# 调用GPU
torch.cuda.set_device(0)
import test_if_cuda_ok
test_if_cuda_ok.test_gpu()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('USE cuda')
else:
    print('USE CPU')
test_model.to(device)

def splicing(images: list, row, column, img_size: list, gap=1):
    if row * column != len(images):
        raise Exception('Img number is not match with row * column')
    height = img_size[0]
    width = img_size[1]
    # 创建空图像
    target = np.zeros(((height+gap) * row, (width+gap) * column), np.uint8)
    target.fill(200)
    # splicing images
    for i in range(row):
        for j in range(column):
            target[i*(height + gap): (i+1)*height+i*gap, j*(width+gap): (j+1)*width+j*gap] = \
                images[i * row + j].copy()
    return target

def test(data):
    result = []
    features = []
    with torch.no_grad():
        #[inputs, labels] = data
        #inputs, labels = inputs.to(device), labels.to(device)
        inputs = data.to(device)
        output = test_model(inputs)
        outputs = test_model(inputs)
        _, predicts = torch.max(output, 1)
        result = output.cpu()
        # 中间层输出
        x = test_model.conv1(inputs)
        x = test_model.bn1(x)
        x = test_model.relu(x)
        x = test_model.maxpool(x)
        x = test_model.layer1(x)
        features.append(x.cpu())
        x = test_model.layer2(x)
        features.append(x.cpu())
        x = test_model.layer3(x)
        features.append(x.cpu())
        x = test_model.layer4(x)
        features.append(x.cpu())
        del x
        del predicts
        del inputs
        del output

    # 处理隐藏层图像，拼接为一个大图像
    img_features = []
    for f in features:
        f = f[0,:,:,:]
        print(f.shape)
        maps = []
        for i in range(f.shape[0]):
            map = f[i].detach().numpy()
            print('map shape: ', map.shape)
            maps.append(np.uint8(map * 255 / (np.max(map) - np.min(map))))
        # 选取最小的平方数目，然后补全其余图像
        show_number_sqrt = int(math.sqrt(len(maps))) + 1
        show_number = show_number_sqrt * show_number_sqrt
        # 补全图像个数
        for i in range(show_number - len(maps)):
            blank_img = np.zeros((f.shape[1], f.shape[2]), np.uint8)
            blank_img.fill(0)
            maps.append(blank_img)
        # 拼接图像
        img_features.append(splicing(maps, show_number_sqrt, show_number_sqrt, [f.shape[1], f.shape[2]]))

    # 处理结果
    # print(result[0])
    result = F.softmax(result[0], dim=0)
    # print(result)
    result = result.tolist()
    # print(result)
    result = [[result[i], i] for i in range(len(result))]
    # print(result)
    result = sorted(result)
    result = list(reversed(result))
    # 制作热图
    # 处理隐藏层图像，拼接为一个大图像
    f = features[0][0, :, :, :]
    hot_map = np.zeros_like(f[0].detach().numpy())
    for i in range(f.shape[0]):
        hot_map = hot_map + np.abs(f[i].detach().numpy())
    hot_map = np.log(hot_map)
    img_features[-1] = np.uint8(hot_map * 255 / (np.max(hot_map) - np.min(hot_map)))
    print(img_features[-1])

    return [result, img_features]

# 建立服务器
def socket_service_image():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('10.134.171.252', 6666))
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    print("Wait for Connection.....................")
    # 只接收一个socket
    sock, addr = s.accept()  # addr是一个元组(ip,port)
    exchange_data(sock, addr)
    s.close()

def exchange_data(sock, addr):
    print("Accept connection from {0}".format(addr))  # 查看发送端的ip和端口

    while True:
        img = receive_data(sock)
        # 接收的是cv版本的图像，要转换到pil版本
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = image_transform(img)
        img = img.reshape((1, *img.shape))
        output = test(img)
        print(output)
        send_data(sock, output)
    sock.close()

socket_service_image()