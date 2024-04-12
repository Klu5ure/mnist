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
        for (x, y) in test_data: # x为图像数据，每次循环包含15张图片，x[0]为第一张, y为标签数据，包含15个标签
            outputs = net.forward(x.view(-1, 28*28))  # 通过模型进行前向传播，outputs里面包含15张图片和它们对应的结果，因为test_data设置了15张图片一个批次
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