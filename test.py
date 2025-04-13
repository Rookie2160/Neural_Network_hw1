from data_reader import CIFAR10Loader
from other import one_hot_encode, compute_accuracy, load_model

def test(model_path="neural_hw1/best_model.pkl"):
    data_loader = CIFAR10Loader(root='./cifar-10-batches-py')
    _, _, X_test, y_test = data_loader.load_data()
    X_test /= 255.0
    y_test = one_hot_encode(y_test, 10)
    model = load_model(model_path)
    y_test_pred = model.forward(X_test, training=False)
    accuracy = compute_accuracy(y_test_pred, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test()
