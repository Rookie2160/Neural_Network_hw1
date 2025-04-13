import matplotlib.pyplot as plt
import numpy as np

def plot_loss_accuracy(train_loss, val_accuracy, save_path="neural_hw1/training_curves.png"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def visualize_weights(model, save_path="neural_hw1/weights_visualization.png", grid_shape=None):
    #将每个隐藏神经元对应的权重重构为(32,32,3)的图像进行展示。

    W = model.W1.copy() 
    num_filters = W.shape[1]
    
    if grid_shape is None:
        grid_cols = int(np.ceil(np.sqrt(num_filters)))
        grid_rows = int(np.ceil(num_filters / grid_cols))
    else:
        grid_rows, grid_cols = grid_shape
        
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols, grid_rows))
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            w = W[:, i]
            w_reshaped = w.reshape(32, 32, 3)
            w_min, w_max = w_reshaped.min(), w_reshaped.max()
            w_norm = (w_reshaped - w_min) / (w_max - w_min + 1e-7)
            ax.imshow(w_norm)
            ax.axis("off")
        else:
            ax.axis("off")
    
    plt.suptitle("Visualization of First Layer Weights", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.show()
