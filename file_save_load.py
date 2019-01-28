import pickle
import time
import data_loader
import os
import torch
import re

# 专门用于处理日志的文件
# 如果文件不存在，则创建文件，文件存在，则续写文件，并添加一行表示时间
# 如果renew为True，则覆盖掉原本的文件
class LogFile:
    def __init__(self, log_file_path, renew=False):
        self.date = time.strftime('%c',time.localtime(time.time()))
        self.path = log_file_path
        self.renew = renew
        # 建立model_built_file，保存建立的信息，现在只有分割验证集的顺序需要保存
        self.__open()
        print('open log file : %s' % self.path)
        #写下日期
        self.file.write('************'+ self.date + '************\n')
        self.__close()

    def write(self, s: str):
        self.__open()
        self.file.write(s)
        self.__close()

    def writelines(self, lines):
        self.__open()
        self.file.writelines(lines)
        self.__close()

    def __close(self):
        self.file.close()

    def __open(self):
    # 建立model_built_file，保存建立的信息，现在只有分割验证集的顺序需要保存
        if not self.renew:
            if not os.path.exists(self.path):
                # print('Make file : %s' % self.path)
                self.file = open(self.path, 'w+')
            else:
                # print('Open file : %s' % self.path)
                self.file = open(self.path, 'a')  # 'a'为续写
        else:
            # print('Make file : %s' % self.path)
            self.file = open(self.path, 'w+')

# 专门用于处理信息存储的文件
# 如果文件不存在，则创建文件，文件存在，则读入文件
# 文件写入均覆盖
class InfoFile:
    def __init__(self, info_file_path, index=0):
        self.data = None
        self.path = info_file_path
        if index != 0:
            self.path = info_file_path + str(index)
        if os.path.exists(info_file_path):
            file = open(info_file_path, 'rb')
            print('load %s success' % info_file_path)
            self.data = pickle.load(file)

    def dump(self, info):
        self.data = info
        file = open(self.path, 'wb+')
        pickle.dump(info, file)
        file.close()
        print('%s dump success' % self.path)

    def load(self):
        if not self.data:
            print('No data to load in %s' % self.path)
            return None
        else:
            print('load info from %s success' % self.path)
            return self.data

# 专门用于处理模型读取、存储的类
# 输入为model文件夹的位置，默认为当前文件位置
# 首先建立模型文件夹model ,在 model 下按照时间存储模型.
#
# 如果 rebuild 为 True，则表示需要重新建立模型，这时根据时间，
# 新建一个文件夹。如果该文件夹存在，
# 则新建一个文件夹名为'$(folder name)-new'
#
# 如果 rebuild 为 FALSE，则表示不需要重建。file读取最新时间的文件夹中的最新的model，载入
# 其中最新的 model，并且记录其中的 epoch 。然后继续进行写入 model 。
# 写：
# 在写的过程中，对模型文件
# 进行编号，模型文件的格式为'model_epoch[$epoch].model'，
class ModelFloder:
    def __init__(self, model_file_folder_locate='.',rebuild=False):
        self.date = time.strftime('%F', time.localtime(time.time()))
        self.model_base_folder = os.path.join(model_file_folder_locate, 'model')
        self.rebuild = rebuild
        self.model_folder: str = ''  # model所在文件夹
        self.model_path: str = ''    # 最新保存的或者加载的model所在的位置
        self.epoch = 0
        self.__build_folder()

    # 读取最新建立的一个模型，如果不存在，则返回None
    # 如果model_path不为空的话，那么按照model_path寻找
    # 如果 model_number不为0的话，那么按照model_number寻找，
    # 默认在最近的一个文件夹中寻找（注意，有可能在新建的文件夹中）
    # 后两者如果没有找到则直接（报错）
    def load_model(self, model_path: str='', epoch_number: int=0):
        if not model_path and epoch_number != 0:
            print('You can not both set model_path and model_number')
            return ''
        # 根据model_path load
        elif model_path:
            print('load model from %s' % model_path)
            self.model_path = model_path
            if not os.path.exists(model_path):
                raise Exception('%s does not exist'%
                                model_path)
        # 根据epoch_number load
        elif epoch_number != 0:
            print('load model %d from %s' % (epoch_number, self.model_folder))
            self.model_path = self.__make_path(self.model_folder, epoch_number)
            if not os.path.exists(self.model_path):
                raise Exception('epoch %d for model does not exist'%
                                epoch_number)
        # 默认方式load
        if self.model_path:
            print('load model success, from %s' % model_path)
            return torch.load(self.model_path)
        else:
            print('model does not exist')
            return ''

    # 保存模型，epoch加1
    def save_model(self, model_object):
        #self.model_path = self.__make_path(self.model_folder, self.epoch)
        torch.save(model_object, self.__make_path(self.model_folder, self.epoch))
        self.epoch = self.epoch + 1
        print('save model success')

    # 建立文件夹，同时初始化epoch，初始化self.init_model_path
    def __build_folder(self):
        # 建立model根文件夹
        if not os.path.exists(self.model_base_folder):
            print('Make model folder. Dir : %s' % self.model_base_folder)
            os.makedirs(self.model_base_folder)
        # 建立存储model的文件夹
        if not self.rebuild:
            # 截取所有文件夹
            folders = [os.path.join(self.model_base_folder,folder)
                       for folder in os.listdir(self.model_base_folder)]
            folders = sorted([folder for folder in folders if os.path.isdir(folder)])
            # 先确定epoch和init model_path
            if not folders:
                self.epoch = 0
                self.model_path = ''
            else:
                if 'ver' not in folders[-1] and \
                        folders[-1] > os.path.join(self.model_base_folder, self.date):
                    raise Exception('folder Wrong format %d', folders[-1])
                # 寻找命名上最后一个folder，在其中寻找epoch进行赋值
                self.epoch = self.__find_epoch(folders[-1]) + 1
                if self.epoch == 0:
                    self.model_path = ''
                else:
                    self.model_path = self.__make_path(folders[-1], self.epoch - 1)

            # 如果文件夹一个都不存在，则建立文件夹
            if not folders:
                self.model_folder = os.path.join(self.model_base_folder, self.date)
                print('Make new folder. Dir : %s' % self.model_folder)
                os.mkdir(self.model_folder)
            # 如果存在，那么直接根据日期是否重复  选择  是否建立新的文件夹
            else:
                last_folder_date = re.findall('\d+-\d+-\d+', folders[-1])[0]  # last folder的创建时间
                if not last_folder_date:
                    raise Exception('last_folder_date can not find, please check date format')
                if last_folder_date != self.date:
                    self.model_folder = os.path.join(self.model_base_folder, self.date)
                    print('Make new folder. Dir : %s' % self.model_folder)
                    os.mkdir(self.model_folder)
                else:
                    self.model_folder = folders[-1]

        else:       # 需要重建，使用__make_new_dir创建文件夹
            folders = [os.path.join(self.model_base_folder,folder)
                       for folder in os.listdir(self.model_base_folder)]
            folders = sorted([folder for folder in folders if os.path.isdir(folder)])
            if not folders:
                self.model_folder = os.path.join(self.model_base_folder, self.date)
                print('Make new folder. Dir : %s' % self.model_folder)
            else:
                self.model_folder = self.__make_new_dir(folders[-1])
                self.epoch = 0

    # 输入最后一个文件夹
    # 根据日期建立文件夹，如果日期重复，寻找'ver%d'，获得版本
    # 返回新建的文件夹的名字
    def __make_new_dir(self, last_folder) -> str:
        new_folder = os.path.join(self.model_base_folder, self.date)
        if not os.path.isdir(last_folder):
            raise Exception('This is not a folder')
        last_folder_date = re.findall('\d+-\d+-\d+', last_folder) # last folder的创建时间
        if not last_folder_date:
            raise Exception('last_folder_date can not find, please check date format')
        last_folder_date = last_folder_date[0]
        # last folder的版本
        last_folder_ver = re.findall('ver\d+', last_folder)
        if not last_folder_ver:
            last_folder_ver = 0
        else:
            last_folder_ver = int(last_folder_ver[0][3:])
        # 如果文件夹重复
        if last_folder_date == self.date:
            new_folder = os.path.join(self.model_base_folder, self.date) \
                            + '-ver%d' % (last_folder_ver + 1)
        print('Make new folder. Dir : %s' % new_folder)
        os.mkdir(new_folder)
        return new_folder

    # 寻找 epoch，如果没有的话，就返回-1
    def __find_epoch(self, folder):
        models = sorted(os.listdir(folder))
        models = [os.path.join(folder, model) for model in models]
        if not models:
            epoch = -1
        else:
            last_epoch = int(re.findall('epoch\d+', models[-1])[0][5:])
            epoch = last_epoch
        return epoch

    def __make_path(self, folder: str, epoch: int):
        if epoch >= 10:
            return os.path.join(folder, 'model_epoch%d.model' % epoch)
        else:
            return os.path.join(folder, 'model_epoch0%d.model' % epoch)

if __name__ == '__main__':
    import build_network
    # model = build_network.basic_model
    # mf = ModelFloder(rebuild=False)
    # mf.date = '2018-12-15'
    # mf.load_model()
    # mf.save_model(model)
    # a = 1
    data