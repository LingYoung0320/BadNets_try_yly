# -*- coding: UTF-8 -*- #
"""
@filename:main_clean.py
@author:Young
@time:2023-12-23
"""
# 负责得到原始干净模型的相关数据

# 加载必要的库
import torch
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据
import torch.nn as nn # “nn”为代号
import torch.nn.functional as F
import torch.optim as optim # 优化器
from torchvision import transforms

# 定义超参数
# 参数 - 未知量
# 超参数 - 自定义的参数，可用来优化策略和模型
BATCH_SIZE = 128 # 每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 决定用GPU还是CPU训练
EPOCH = 2 # 训练数据集的轮次

# 构建pipeline/transforms（对图像做变换）
pipeline = transforms.Compose([
    transforms.ToTensor(), # 将图片转换成tensor，数值型的容器
    transforms.Normalize((0.1307,),(0.3081,)) # 正则化：降低模型复杂度，防止过拟合（过拟合：模型只认识训练过的，对未见过的测试集表现不佳）
])

# 使用内置函数下载 mnist 数据集
from torch.utils.data import DataLoader # 数据处理的库
train_set = mnist.MNIST('./data', train=True, download=False, transform=pipeline)
test_set = mnist.MNIST('./data', train=False, download=False, transform=pipeline)

# 查看数据集
# a_data, a_label = train_set[0]
# a_data.show()
# print(a_label)

# 加载数据集
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # shuffle=True打乱数据
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# 构建网络模型
# 继承父类，构造，调用父类
class Digit(nn.Module):
    def __init__(self, input_channels, output_num):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        fc1_input_features = 800 if input_channels == 3 else 512
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc1_input_features, out_features=512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=output_num),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 定义优化器
model = Digit(input_channels=1, output_num=10).to(DEVICE)

optimizer=optim.Adam(model.parameters()) # 选择adam优化器

# 定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
    #模型训练
    model.train()
    for batch_index,(data, target) in enumerate(train_loader): # data：数据 target：标签
        # 部署到DEVICE上去
        data,target=data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 预测,训练后结果
        output=model(data)
        # 计算损失（将预测结果和实际标签做计算）
        loss = F.cross_entropy(output, target)#多分类用交叉验证
        # 找到概率最大的下标
        # pred = output.argmax(1)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index%3000==0:
            print("Epoch {}  \t Loss {:.6f}".format(epoch,loss.item()))

# 定义测试方法
def tst_model(model, device, test_loader):
    #模型验证
    model.eval()
    #正确率
    corrcet=0.0
    #测试损失
    test_loss=0.0
    with torch.no_grad():   #不会计算梯度，也不会进行反向传播
            for data,target in test_loader:
                #部署到device上
                data,target=data.to(device),target.to(device)
                #测试数据
                output=model(data)
                #计算测试损失
                test_loss+=F.cross_entropy(output,target).item()
                #找到概率最大的下标
                pred=output.argmax(1)
                #累计正确的值
                corrcet+=pred.eq(target.view_as(pred)).sum().item()
            test_loss/=len(test_loader.dataset)
            print("TestLoss {:.4f} Accuracy {:.3f}\n".format(test_loss,100.0 * corrcet / len(test_loader.dataset)))

# 调用训练方法和测试方法开始训练，并输出预测结果
for epoch in range(1, EPOCH+1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    tst_model(model, DEVICE, test_loader)

# 输出每个数字的准确率
def each_tst_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = [0] * 10  # 初始化每个数字的正确计数
    total = [0] * 10    # 初始化每个数字的总计数
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():   # 不会计算梯度，也不会进行反向传播
        for data, target in test_loader:
            # 部署到device上
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率最大的下标
            pred = output.argmax(1)
            # 累计正确的值
            correct_tensor = pred.eq(target.view_as(pred))
            for i in range(10):
                correct[i] += correct_tensor[target == i].sum().item()
                total[i] += target[target == i].size(0)

        test_loss /= len(test_loader.dataset)
        print("Test Loss: {:.4f}".format(test_loss))
        for i in range(10):
            accuracy = 100.0 * correct[i] / total[i]
            print("Error for digit {}: {:.3f}% (total: {})".format(i, 100-accuracy, total[i]))

# 输出
each_tst_model(model, DEVICE, test_loader)