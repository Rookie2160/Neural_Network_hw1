# Neural_Network_hw1

作业：从零开始构建三层神经网络分类器，实现图像分类

基本要求：
（1） 本次作业要求自主实现反向传播，不允许使用 pytorch，tensorflow 等现成的支持自动微分的深度学习框架，可以使用 numpy；
（2） 最终提交的代码中应至少包含模型、训练、测试和参数查找四个部分，鼓励进行模块化设计；
（3） 其中模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；训练部分应实现 SGD 优化器、学习率下降、交叉熵损失和 L2 正则化，并能根据验证集指标自动保存最优的模型权重；参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）。
## 项目介绍

```
project
├── nnmodel.py        # 定义神经网络类（含前向、反向传播、loss计算）
├── data_reader.py    # 实现 CIFAR-10 数据加载
├── train.py          # 实现训练过程、记录loss和验证accuracy，并可视化
├── test.py           # 用于加载保存的模型，对测试数据进行评估
├── other.py          # 辅助函数，如计算准确率、保存/加载模型、one-hot编码等
├── visualize.py      # 绘图
└── main.py           # 主程序，整合数据加载、训练、测试流程，运行该文件即可
```


## 数据准备

请下载 CIFAR-10 数据集并将其解压到项目目录下的`cifar-10-batches-py`
文件夹中。

链接https://www.cs.toronto.edu/~kriz/cifar.html

## 训练模型

打开终端或命令提示符，进入项目所在目录，然后运行：

```
python main.py
```

训练过程中，每 100 次迭代会打印一次当前训练损失值。训练结束后，最佳模型参数会保存到 `best_model.pkl`

同时训练过程中的曲线图将保存在 `training_curves.png`

第一层权重可视化图保存在 `weights_visualization.png`

## 测试模型

训练结束后，代码会自动加载最佳模型，并在测试集上测试打印最终准确率。

你也可以单独运行：

```
python test.py
```

## 模型权重

通过网盘分享的文件：best_model.pkl

链接: https://pan.baidu.com/s/18AMuifBePQh3z7ZF39iiyQ?pwd=9239 
提取码: 9239
