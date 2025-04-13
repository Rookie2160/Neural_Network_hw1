import numpy as np
import pickle
import os

class CIFAR10Loader:
    def __init__(self, root):
        self.root = root
    
    def _load_batch(self, filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X = datadict[b'data']
            Y = datadict[b'labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y

    def _load_train_data(self):
        xs, ys = [], []
        for b in range(1, 6):
            batch_file = os.path.join(self.root, f'data_batch_{b}')
            X, Y = self._load_batch(batch_file)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        return Xtr, Ytr

    def _load_test_data(self):
        test_file = os.path.join(self.root, 'test_batch')
        return self._load_batch(test_file)
    
    def load_data(self):
        Xtr, Ytr = self._load_train_data()
        Xte, Yte = self._load_test_data()
        return Xtr, Ytr, Xte, Yte
