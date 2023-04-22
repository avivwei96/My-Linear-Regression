from copy import deepcopy
import torch
from tqdm import tqdm


class MyLinearRegression(torch.nn.Module):

    def __init__(self, input_dim, lr=0.001):
        super(MyLinearRegression, self).__init__()
        self.lr = lr
        self.w = torch.rand(size=(input_dim + 1, 1), dtype=torch.float64)

    def forward(self, x):
        # this function predict real number for a matrix of the data
        ones = torch.ones((x.shape[0], 1))
        x_temp = torch.cat((x, ones), dim=1)
        y_pred = []
        for i in range(x.numel()):
            y_pred.append(x_temp[i] @ self.w)
        return y_pred

    def update_weights(self, weights_derv):
        # this function update the weights
        self.w = self.w - weights_derv.reshape(self.w.shape) * self.lr


    def get_weights(self):
        # this function return the weight of the model
        return deepcopy(self.w)

    def MSELoss(self, X, targets):
        # this function calculates the loss of the model
        outputs =self.forward(X)
        loss_lst = [(targets[i] - outputs[i]) ** 2 for i in range(len(targets))]
        loss = sum(loss_lst) / len(loss_lst)
        return loss

    def MSELossDerv(self, x, targets):
        # this function calculate the drev of the loss
        ones = torch.ones((x.shape[0], 1))
        x_temp = torch.cat((x, ones), dim=1)
        outputs = self.forward(x)
        # creates a list of all the losses
        loss_derv_lst = [-1 * (targets[i] - outputs[i]) * (x_temp[i]) for i in range(len(targets))]
        self.loss_derv = torch.tensor([0., 0.])

        # calculate the loss derv
        for i in range(len(loss_derv_lst)):
            self.loss_derv += loss_derv_lst[i]

        self.loss_derv = (self.loss_derv) * 2 / len(targets)
        return self.loss_derv

    def train(self, dataloader, n_epochs):
        history_loss = []
        history_lines = []
        for epoch in tqdm(range(n_epochs)):
            for x, targets in dataloader:
                loss = self.MSELoss(x, targets)
                loss_derv = self.MSELossDerv(x, targets)
                self.update_weights(loss_derv)