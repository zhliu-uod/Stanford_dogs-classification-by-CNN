# coding: utf-8
# ------------------------------------ step 1/6 : 导入必要的包以及初始化定义 ----------------------------------------------------
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append("..")
from PIL import Image
from torch.utils.data import Dataset
import time
from torchvision import models

# 标签的路径，包含图片的路径和类别
train_txt_path = os.path.join("/train.txt")  # 把制作好的训练集标签路径放进去
valid_txt_path = os.path.join("/valid.txt")  # 把制作好的测试集标签路径放进去
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
               '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
               '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
               '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
               '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
               '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
               '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
               '71', '72', '73', '74', '75', '76', '77', '78', '79', '80',
               '81', '82', '83', '84', '85', '86', '87', '88', '89', '90',
               '91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
               '101', '102', '103', '104', '105', '106', '107', '108', '109', '110',
               '111', '112', '113', '114', '115', '116', '117', '118', '119']

labels_to_classes = {
    """
    这里需要加入自己的最终预测对应字典，例如：
      '0': '花'
    """
    '0': 'Chihuahua', '1': 'Japanese_spaniel', '2': 'Maltese_dog', '3': 'Pekinese', '4': 'Shih-Tzu',
    '5': 'Blenheim_soaniel', '6': 'papillon', '7': 'toy_terrier', '8': 'Rhodesian_ridgeback', '9': 'Afghan_hound',
    '10': 'basset', '11': 'beagle', '12': 'bloodhound', '13': 'bluetick', '14': 'black-and-tan_coonhound',
    '15': 'Walker_hound', '16': 'English_foxhound', '17': 'redbone', '18': 'borzoi', '19': 'Irish_wolfhound',
    '20': 'Italian_greyhound', '21': 'whippet', '22': 'Ibizan_hound', '23': 'Norwegian_elkhound', '24': 'otterhound',
    '25': 'Saluki', '26': 'Scottish_deerhound', '27': 'Weimaraner', '28': 'Staffordshire_bullterrier',
    '29': 'American_Staffordshire_terrier',
    '30': 'Bedlington_terrier', '31': 'Border_terrier', '32': 'Kerry_blue_terrier', '33': 'Irish_terrier',
    '34': 'Norfolk_terrier',
    '35': 'Norwich_terrier', '36': 'Yorkshire_terrier', '37': 'wire-haired_fox_terrier', '38': 'Lakeland_terrier',
    '39': 'Sealyham_terrier',
    '40': 'Airedale', '41': 'cairn', '42': 'Australian_terrier', '43': 'Dandie_Dinmont', '44': 'Boston_bull',
    '45': 'miniature_schnauzer', '46': 'giant_schnauzer', '47': 'standard_schnauzer', '48': 'Scotch_terrier',
    '49': 'Tibetan_terrier',
    '50': 'silky_terrier', '51': 'soft-coated_wheaten_terrier', '52': 'West_Highland_white_terrier', '53': 'Lhasa',
    '54': 'flat-coated_retriever',
    '55': 'curly-coated_retriever', '56': 'golden_retriever', '57': 'Labrador_retriever',
    '58': 'Chesapeake_Bay_retriever', '59': 'German_short-haired_pointer',
    '60': 'vizsla', '61': 'English_setter', '62': 'Irish_setter', '63': 'Gordon_setter', '64': 'Brittany_spaniel',
    '65': 'clumber', '66': 'English_springer', '67': 'Welsh_springer_spaniel', '68': 'cocker_spaniel',
    '69': 'Sussex_spaniel',
    '70': 'Irish_water_spaniel', '71': 'kuvasz', '72': 'schipperke', '73': 'groenendael', '74': 'malinois',
    '75': 'briard', '76': 'kelpie', '77': 'komondor', '78': 'Old_English_sheepdog', '79': 'Shetland_sheepdog',
    '80': 'collie', '81': 'Border_collie', '82': 'Bouvier_des_Flandres', '83': 'Rottweiler', '84': 'German_shepherd',
    '85': 'Doberman', '86': 'miniature_pinscher', '87': 'Greater_Swiss_Mountain_dog', '88': 'Bernese_mountain_dog',
    '89': 'Appenzeller',
    '90': 'EntleBucher', '91': 'boxer', '92': 'bull_mastiff', '93': 'Tibetan_mastiff', '94': 'French_bulldog',
    '95': 'Great_Dane', '96': 'Saint_Bernard', '97': 'Eskimo_dog', '98': 'malamute', '99': 'Siberian_husky',
    '100': 'affenpinscher', '101': 'basenji', '102': 'pug', '103': 'Leonberg', '104': 'Newfoundland',
    '105': 'Great_Pyrenees', '106': 'Samoyed', '107': 'Pomeranian', '108': 'chow', '109': 'keeshond',
    '110': 'Brabancon_griffon', '111': 'Pembroke', '112': 'Cardigan', '113': 'toy_poodle', '114': 'miniature_poodle',
    '115': 'standard_poodle', '116': 'Mexican_hairless', '117': 'dingo', '118': 'dhole', '119': 'African_hunting_dog',

}
train_bs = 16  # 一次喂入训练中的样本量
valid_bs = 1  # 一次喂入验证中的样本量
lr_init = 1e-6  # 初始学习率
max_epoch = 500  # 最大训练轮数

print(torch.cuda.is_available())
use_gpu = torch.cuda.is_available()

# log

result_dir = os.path.join("..", "..", "Result")
time_open = time.time()

# ------------------------------------ step 2/6 : 加载数据 -------------------------------------------------------------------
# 数据预处理设置

normMean = [0.4760485, 0.45188636, 0.39048073]  # 通过样本计算得到的均值
normStd = [0.26165572, 0.25607935, 0.26033863]  # 通过样本计算得到的标准差

normTransform = transforms.Normalize(normMean, normStd)  # 归一化处理

trainTransform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    normTransform
])

validTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)

# MyDataset实例化
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)


# 数据预览
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.4760485, 0.45188636, 0.39048073])
    std = np.array([0.26165572, 0.25607935, 0.26033863])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(3)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

# ------------------------------------ step 3/6 : 定义网络 --------------------------------------------------------------------
# 搭建模型结构
'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        # self.dropout1 = nn.Dropout()
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout()


        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout()

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout()

        self.fc1 = nn.Linear(256*12*12, 1000)
        # self.dropout5 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(1000, 1000)
        # self.dropout6 = nn.Dropout(p = 0.5)
        self.fc3 = nn.Linear(1000, 250)
        # self.dropout7 = nn.Dropout(p = 0.5)
        self.fc4 = nn.Linear(250, 120)

    def forward(self, x):

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        #flatten
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

    # 定义权值初始化

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
'''
# fine-tune vgg16
net = models.vgg16(pretrained=True)
print(net)

for parma in net.parameters():
    parma.requires_grad = False

net.classifier = torch.nn.Sequential(
    nn.Linear(25088, 1000),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1000, 250),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(250, 120))

# net = Net()     # 创建一个网络
if use_gpu:
    net = net.cuda()
    
# print(net)

# net.initialize_weights()  # 初始化权值

# ------------------------------------ step 4/6 : 定义损失函数和优化器 ----------------------------------------------------------

criterion = nn.CrossEntropyLoss()  # 选择损失函数
# optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    # 选择优化器
optimizer = optim.Adam(net.parameters(), lr=lr_init)  # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略

# ------------------------------------ step 5/6 : 训练 ------------------------------------------------------------------------

for epoch in range(max_epoch):

    loss_sigma = 0.0  # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率
    print("Epoch {}/{}".format(epoch, max_epoch))
    for data in train_loader:
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().cpu().numpy()
        loss_sigma = 0.0
        loss_sigma += loss.item()

    print("train loss: {:.4f} train accuracy: {:.2%}".format(loss_sigma, correct / total))
    # ------------------------------------ step 6/6 : 验证 ---------------------------------------------------------------------
    # 调成训练模式
    net.eval()
    for data in valid_loader:

        # 获取图片和标签
        images, labels = data
        if use_gpu:
            inputs, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(images), Variable(labels)

        total += labels.size(0)

        # forward
        outputs = net(inputs)
        if use_gpu:
            outputs = Variable(outputs.cuda())
        else:
            outputs = outputs

        # 计算loss
        loss = criterion(outputs, labels)
        loss_sigma += loss.item()

        # 统计
        _, predicted = torch.max(outputs.data, 1)
        # labels = labels.data    # Variable --> tensor
        correct += (predicted == labels).squeeze().sum().cpu().numpy()

    print('valid accuracy:{:.4f} valid accuracy:{:.2%}'.format(loss_sigma, correct / total))

time_end = time.time() - time_open
print('total_time = :',time_end)
