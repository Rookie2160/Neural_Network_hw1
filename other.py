import numpy as np
import pickle
import os
from nnmodel import NeuralNetwork

def compute_accuracy(y_pred, y_true):
    predicted_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(predicted_labels == true_labels)

def save_model(model, filepath):
    model_dict = {
        'W1': model.W1,
        'b1': model.b1,
        'W2': model.W2,
        'b2': model.b2,
        'W3': model.W3,
        'b3': model.b3,
        'activation_function': model.activation_function,
        'lambda_reg': model.lambda_reg,
        'input_size': model.input_size,
        'hidden_sizes': model.hidden_sizes,
        'output_size': model.output_size,
        'dropout_rate': model.dropout_rate,
        'momentum': model.momentum
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    hidden_sizes = model_data.get('hidden_sizes', [model_data['W1'].shape[1]])
    model = NeuralNetwork(
        input_size=model_data['input_size'],
        hidden_sizes=hidden_sizes,
        output_size=model_data['output_size'],
        activation_function=model_data['activation_function'],
        lambda_reg=model_data['lambda_reg'],
        dropout_rate=model_data.get('dropout_rate', 0.0),
        momentum=model_data.get('momentum', 0.9)
    )
    model.W1 = model_data['W1']
    model.b1 = model_data['b1']
    model.W2 = model_data['W2']
    model.b2 = model_data['b2']
    model.W3 = model_data['W3']
    model.b3 = model_data['b3']
    return model

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y.reshape(-1)]

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
