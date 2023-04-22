# this main is an example of how to use the model
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
from MyLinearRegression import MyLinearRegression


class LinearDataset(Dataset):
    """
    A dataset returning x and y using the linear equation.
    The x values should be sample from 0-1 uniformly.
    """

    def __init__(self, num_samples, m, n):
        """
        :param num_samples: Number of samples (labeled images in the dataset)
        :param m: The ground truth affline transformation
        :param n: The ground truth bais
        """
        super().__init__()
        self.num_samples = num_samples
        self.m = m
        self.n = n
        self.X = np.random.rand(self.num_samples, len(self.m))
        self.Y = []
        for var in self.X:
            a = np.random.normal(loc=0, scale=0.5)
            self.Y.append(a + self.n + self.m @ var)

        self.dataset = list(zip(self.X, self.Y))

    def __getitem__(self, index):
        sample = self.dataset[index][0]
        label = self.dataset[index][1]

        return sample, label

    def __len__(self):
        return len(self.dataset)



if __name__ == '__main__':
    linear_ds = LinearDataset(100, np.array([3]), np.array([4]))
    X = []
    y = []
    for _x, _y in linear_ds:
        X.append(_x)
        y.append(_y)
    plt.scatter(X, y)
    line_x = np.linspace(0, 1., 100)
    plt.plot(line_x, 4 + 3 * line_x, c='red')
    plt.show()

    linear_dl = torch.utils.data.DataLoader(linear_ds, batch_size=100, shuffle=True)

    linear_regression_model = MyLinearRegression(1, 0.01)
    linear_regression_model.train(linear_dl, 1000)

    convergenced_w, convergenced_b = linear_regression_model.get_weights()[0][0], linear_regression_model.get_weights()[1][0]

    print(f"Model convergence to the following weights: \nw = {convergenced_w}, b = {convergenced_b}")

    plt.scatter(X, y)
    line_x = np.linspace(0, 1., 100)
    intercept = convergenced_b
    slope = convergenced_w
    plt.plot(line_x, intercept + slope * line_x, c='red')
    plt.show()