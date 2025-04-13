import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation_function='relu', lambda_reg=0.0, dropout_rate=0.0, momentum=0.9):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_function = activation_function
        self.lambda_reg = lambda_reg
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        
        # 初始化各层权重和偏置
        self.W1 = np.random.randn(input_size, hidden_sizes[0]) * 0.01
        self.b1 = np.zeros((1, hidden_sizes[0]))
        
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * 0.01
        self.b2 = np.zeros((1, hidden_sizes[1]))
        
        self.W3 = np.random.randn(hidden_sizes[1], output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
        
        # 初始化Momentum变量
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)
    
    def activate(self, X):
        if self.activation_function == "relu":
            return np.maximum(0, X)
        elif self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-X))
        elif self.activation_function == "tanh":
            return np.tanh(X)
        else:
            raise ValueError("Unsupported activation function")
    
    def activate_deriv(self, A, Z):
        if self.activation_function == "relu":
            return (Z > 0).astype(float)
        elif self.activation_function == "sigmoid":
            return A * (1 - A)
        elif self.activation_function == "tanh":
            return 1 - np.square(A)
        else:
            raise ValueError("Unsupported activation function")
    
    def dropout(self, A):
        if self.dropout_rate > 0:
            mask = (np.random.rand(*A.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            return A * mask, mask
        else:
            return A, None
    
    def softmax(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def forward(self, X, training=True):
        if X.ndim != 2:
            X = X.reshape(X.shape[0], -1)
        self.X = X
        # 第一隐藏层
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activate(self.Z1)
        if training:
            self.A1, self.dropout_mask1 = self.dropout(self.A1)
        else:
            self.dropout_mask1 = None
        
        # 第二隐藏层
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.activate(self.Z2)
        if training:
            self.A2, self.dropout_mask2 = self.dropout(self.A2)
        else:
            self.dropout_mask2 = None
        
        # 输出层
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3
    
    def compute_loss(self, X, y):
        m = X.shape[0]
        probs = self.forward(X, training=False)
        correct_logprobs = -np.log(probs[range(m), y.argmax(axis=1)] + 1e-7)
        data_loss = np.sum(correct_logprobs) / m
        reg_loss = 0.5 * self.lambda_reg * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2))
        return data_loss + reg_loss
    
    def backward(self, X, y, learning_rate):
        if X.ndim != 2:
            X = X.reshape(X.shape[0], -1)
        m = X.shape[0]
        # 调用forward，利用存储值反向传播
        dZ3 = self.A3 - y 
        dW3 = np.dot(self.A2.T, dZ3) / m + (self.lambda_reg / m) * self.W3
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        dA2 = np.dot(dZ3, self.W3.T)
        if self.dropout_mask2 is not None:
            dA2 *= self.dropout_mask2
        dZ2 = dA2 * self.activate_deriv(self.A2, self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m + (self.lambda_reg / m) * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        if self.dropout_mask1 is not None:
            dA1 *= self.dropout_mask1
        dZ1 = dA1 * self.activate_deriv(self.A1, self.Z1)
        dW1 = np.dot(self.X.T, dZ1) / m + (self.lambda_reg / m) * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Momentum更新参数
        self.vW3 = self.momentum * self.vW3 - learning_rate * dW3
        self.vb3 = self.momentum * self.vb3 - learning_rate * db3
        self.W3 += self.vW3
        self.b3 += self.vb3
        
        self.vW2 = self.momentum * self.vW2 - learning_rate * dW2
        self.vb2 = self.momentum * self.vb2 - learning_rate * db2
        self.W2 += self.vW2
        self.b2 += self.vb2
        
        self.vW1 = self.momentum * self.vW1 - learning_rate * dW1
        self.vb1 = self.momentum * self.vb1 - learning_rate * db1
        self.W1 += self.vW1
        self.b1 += self.vb1
