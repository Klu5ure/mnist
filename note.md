# 学习笔记：使用PyTorch构建简单的神经网络进行MNIST手写数字识别

## 概述

在这份学习笔记中，我们将通过一个实际的Python代码示例，了解如何使用PyTorch库构建一个简单的神经网络模型，并用它来识别MNIST数据集中的手写数字。

## 环境准备

在开始之前，确保已经安装了PyTorch库。如果没有安装，可以通过以下命令进行安装：

```bash
pip install torch torchvision
```

## 代码解析

### 1. 导入必要的库

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
```

这部分代码导入了我们需要的库，包括PyTorch的核心库、数据加载器、图像转换工具、MNIST数据集以及绘图库matplotlib。

### 2. 定义神经网络模型

```python
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x
```

这里定义了一个名为`Net`的类，它继承自`torch.nn.Module`。在这个类中，我们定义了四个全连接层（`fc1`到`fc4`），其中`fc1`将输入的图像数据（经过展平）映射到64维的特征空间，接下来的三个全连接层都保持64维的特征空间，最后一个全连接层将特征映射到10维，对应于10个类别（0到9的数字）。
`forward`方法定义了数据通过网络的前向传播过程，首先通过ReLU激活函数进行非线性变换，最后一层使用log_softmax函数进行输出，这在多分类问题中常用于计算交叉熵损失。

### 3. 获取数据加载器

```python
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True) # 15张图片为一批
```

这个函数用于获取训练集和测试集的数据加载器。
`MNIST`函数用于下载并加载数据集，`DataLoader`则用于批量加载数据，并在每个epoch中随机打乱数据顺序。
MNIST是一个广泛使用的入门级数据集，包含了大量的手写数字图像。

### 4. 评估模型性能

```python
# 定义一个函数，用于评估模型在测试集上的准确率
def evaluate(test_data, net):
    n_correct = 0  # 正确预测的数量
    n_total = 0    # 总预测数量
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        for (x, y) in test_data: # test_data一共4000份数据。每份数据包含x和y。x为图像数据，包含15张图片，x[0]为第一张。y为标签数据，包含15个标签
            outputs = net.forward(x.view(-1, 28*28))  # 通过模型进行前向传播，outputs里面包含15张图片的预测结果
            for i, output in enumerate(outputs): # 遍历outputs的15个结果，与y中的15个真实值一一比对
                if torch.argmax(output) == y[i]: # output是一张图片的结果，y[i]是这张图片的真实值
                    n_correct += 1  # 如果预测正确，则增加正确数量
                n_total += 1  # 增加总预测数量
    return n_correct / n_total  # 返回准确率
```

`evaluate`函数用于计算模型在测试集上的准确率。它通过前向传播得到模型的预测输出，然后与真实标签进行比较，统计正确预测的数量，并计算准确率。

##### **关于test_data**

遍历测试集，看看里面是什么内容

```python
for (x, y) in test_data:
    print(x)
    print(y)
    break
tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        ...,


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]]])
tensor([4, 3, 3, 9, 9, 8, 1, 7, 3, 3, 2, 5, 7, 8, 4])



for (x, y) in test_data:
    print(x.size())
    break
torch.Size([15, 1, 28, 28])
```

x是一个张量，[15, 1, 28, 28]，15代表批次，也就是一次15张图片，1代表单通道，两个28代表图片宽高

y代表的是这十五张手写数字图片所对应的真实数字

##### 关于张量

```json
# (2, 3, 2, 2) 2张图片，三通道，2x2像素如下
[
    [
        [
            [2,4],
            [6,9]
        ],
        [
            [0,7],
            [4,5]
        ],
        [
            [2,4],
            [9,9]
        ]
    ],
    [
        [
            [0,0],
            [3,8]
        ],
        [
            [9,0],
            [2,9]
        ],
        [
            [3,0],
            [3,6]
        ]
    ]
]

# (2, 1, 2, 2) 两张图片，单通道，2x2像素
[
    [
        [
            [9,0],
            [3,6]
        ]
    ],
    [
        [
            [3,3],
            [3,4]
        ]
    ]
]
```

### 5. 主函数

```python
def main():
    train_data = get_data_loader(is_train=True)  # 获取训练数据加载器
    test_data = get_data_loader(is_train=False)  # 获取测试数据加载器
    net = Net()  # 实例化神经网络

    # 打印模型在没有任何训练之前的准确率
    print("initial accuracy:", evaluate(test_data, net))
    # 定义优化器，这里使用Adam算法
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # 训练模型，循环2个epoch
    for epoch in range(2):
        # 遍历训练数据集中的每个批次
        for (x, y) in train_data:
            net.zero_grad()  # 清空之前的梯度
            output = net.forward(x.view(-1, 28*28))  # 计算模型输出
            loss = torch.nn.functional.nll_loss(output, y)  # 计算损失函数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
        # 每个epoch结束后，打印当前的准确率
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    # 展示几个测试图像及其预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28), cmap='gray')  # 显示图像，使用灰度色图
        plt.title("prediction: " + str(int(predict)))
    plt.show()  # 显示图像

```

`main`函数是程序的主要执行部分。它首先获取训练和测试数据加载器，初始化神经网络模型，然后通过多个epoch进行训练，每个epoch结束后打印当前的准确率。
训练完成后，它还会展示几个测试图像及其预测结果。

## 完整代码和注释

```python
# 导入所需的库
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 定义一个神经网络类，继承自torch.nn.Module
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法
        # 定义四个全连接层
        self.fc1 = torch.nn.Linear(28*28, 64)  # 输入层到隐藏层，输入特征数为28*28，输出特征数为64
        self.fc2 = torch.nn.Linear(64, 64)    # 隐藏层到另一个隐藏层，特征数保持64
        self.fc3 = torch.nn.Linear(64, 64)    # 同上
        self.fc4 = torch.nn.Linear(64, 10)   # 最后一个隐藏层到输出层，输出特征数为10，对应10个类别

    def forward(self, x):
        # 定义数据如何通过网络的前向传播过程
        x = torch.nn.functional.relu(self.fc1(x))  # 第一个隐藏层的激活函数，使用ReLU
        x = torch.nn.functional.relu(self.fc2(x))  # 第二个隐藏层的激活函数，使用ReLU
        x = torch.nn.functional.relu(self.fc3(x))  # 第三个隐藏层的激活函数，使用ReLU
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)  # 输出层的激活函数，使用log_softmax
        return x  # 返回网络的输出

# 定义一个函数，用于获取数据加载器
def get_data_loader(is_train):
    # 定义图像预处理操作，将图像转换为张量
    to_tensor = transforms.Compose([transforms.ToTensor()])
    # 下载并加载MNIST数据集，根据is_train参数决定是训练集还是测试集
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    # 创建DataLoader，用于批量加载数据，并在每个epoch中随机打乱数据顺序
    return DataLoader(data_set, batch_size=15, shuffle=True)

# 定义一个函数，用于评估模型在测试集上的准确率
def evaluate(test_data, net):
    n_correct = 0  # 正确预测的数量
    n_total = 0    # 总预测数量
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))  # 通过模型进行前向传播，outputs里面包含15张图片的结果，因为test_data设置了15张图片一个批次
            for i, output in enumerate(outputs): # 遍历outputs的15个结果，与y中的15个真实值一一比对
                if torch.argmax(output) == y[i]: # output是一张图片的结果，y[i]是这张图片的真实值
                    n_correct += 1  # 如果预测正确，则增加正确数量
                n_total += 1  # 增加总预测数量
    return n_correct / n_total  # 返回准确率

# 定义主函数
def main():
    train_data = get_data_loader(is_train=True)  # 获取训练数据加载器
    test_data = get_data_loader(is_train=False)  # 获取测试数据加载器
    net = Net()  # 实例化神经网络

    # 打印模型在没有任何训练之前的准确率
    print("initial accuracy:", evaluate(test_data, net))
    # 定义优化器，这里使用Adam算法
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # 训练模型，循环2个epoch
    for epoch in range(2):
        # 遍历训练数据集中的每个批次
        for (x, y) in train_data:
            net.zero_grad()  # 清空之前的梯度
            output = net.forward(x.view(-1, 28*28))  # 计算模型输出
            loss = torch.nn.functional.nll_loss(output, y)  # 计算损失函数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
        # 每个epoch结束后，打印当前的准确率
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    # 展示几个测试图像及其预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28), cmap='gray')  # 显示图像，使用灰度色图
        plt.title("prediction: " + str(int(predict)))
    plt.show()  # 显示图像

# 如果这个脚本是主程序，则运行main函数
if __name__ == "__main__":
    main()
```



## 总结

通过这份学习笔记，我们了解了如何使用PyTorch构建一个简单的神经网络模型，并用MNIST数据集进行训练和测试。这个过程包括定义模型结构、获取数据、评估性能和模型训练等关键步骤。