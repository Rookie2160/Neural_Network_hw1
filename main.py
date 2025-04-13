from data_reader import CIFAR10Loader
from nnmodel import NeuralNetwork
from other import one_hot_encode, ensure_dir
from train import train
from test import test
from visualization import visualize_weights, plot_loss_accuracy

def main():
    data_loader = CIFAR10Loader(root='./cifar-10-batches-py')
    X_train, y_train, X_test, y_test = data_loader.load_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # 数据归一化
    X_train /= 255.0
    X_test /= 255.0
    y_train = one_hot_encode(y_train, 10)
    y_test = one_hot_encode(y_test, 10)
    
    X_val = X_train[:5000]
    y_val = y_train[:5000]
    X_train_split = X_train[5000:]
    y_train_split = y_train[5000:]
    
    # 网络参数设置
    input_size = 32 * 32 * 3       # 3072
    hidden_sizes = [256, 128]      # 两层隐藏层
    output_size = 10
    activation_function = 'relu'
    lambda_reg = 1e-2              # L2 正则系数
    dropout_rate = 0.5             # Dropout 概率
    momentum = 0.9
    learning_rate = 0.01
    epochs = 30
    batch_size = 32
    
    # 创建模型
    model = NeuralNetwork(input_size, hidden_sizes, output_size, activation_function, lambda_reg, dropout_rate, momentum)
    ensure_dir("neural_hw1")
    best_model, train_loss_history, val_accuracy_history = train(model, X_train_split, y_train_split, X_val, y_val,
                                                                 epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, print_every=100)
    
    # 绘制训练曲线
    plot_loss_accuracy(train_loss_history, val_accuracy_history, save_path="neural_hw1/training_curves.png")

    # 可视化第一层权重
    visualize_weights(best_model, save_path="neural_hw1/weights_visualization.png", grid_shape=(16, 16))
    
    # 自动测试最佳模型
    print("Testing the best model on the test set:")
    test("neural_hw1/best_model.pkl")

if __name__ == "__main__":
    main()
