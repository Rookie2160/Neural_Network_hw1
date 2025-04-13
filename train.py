import numpy as np
import matplotlib.pyplot as plt
from other import compute_accuracy, save_model, ensure_dir

def train(model, X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.01, batch_size=32, print_every=100):
    num_train = X_train.shape[0]
    iterations_per_epoch = int(np.ceil(num_train / batch_size))
    total_iterations = epochs * iterations_per_epoch

    train_loss_history = []
    val_accuracy_history = []

    best_val_accuracy = 0
    best_model = None

    iter_count = 0
    for epoch in range(epochs):
        perm = np.random.permutation(num_train)
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        for i in range(0, num_train, batch_size):
            iter_count += 1
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            # 前向传播和计算 loss
            loss = model.compute_loss(X_batch, y_batch)
            if iter_count % print_every == 0:
                print(f"Iteration {iter_count}/{total_iterations}, Loss: {loss:.4f}")
            train_loss_history.append(loss)
            # 反向传播更新参数
            model.backward(X_batch, y_batch, learning_rate)
        
        # 评估准确率
        y_val_pred = model.forward(X_val, training=False)
        val_acc = compute_accuracy(y_val_pred, y_val)
        val_accuracy_history.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} completed, Validation Accuracy: {val_acc:.4f}")
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model = model
            save_model(model, "neural_hw1/best_model.pkl")
    
    # 绘制训练曲线
    ensure_dir("neural_hw1")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracy_history, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("neural_hw1/training_curves.png")
    plt.show()
    
    return best_model, train_loss_history, val_accuracy_history
