import torch
import torch.nn as nn
import numpy as np


class LSTM(nn.Module):
    """
    implemented using LSTM module based on pytorch
    self.layer1 : 神经网络
    self.layer2 : 全连接层
    """
    def __init__(self, input_size=32, hidden_size=20, output_size=1, num_layer=20):
        super(LSTM, self).__init__()
        # 神经网络
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        # 全连接层
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        # print('x: ', x.shape)
        s, b, h = x.size()
        # print('s: ', s)
        # print('b: ', b)
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


def train_model(training_x, training_y, model, episodes=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # print('x: ', training_x.shape)
    training_x = training_x.reshape(-1, 1, 32)
    # print('x: ', training_x.shape)
    # print(training_data_y)
    training_y = np.array(training_y, dtype=np.float32)
    # print('y: ', training_y.shape)
    # print('yes')
    # print(training_data_y)
    training_y = training_y.reshape(-1, 1, 1)
    # print('y: ', training_y)
    var_x = torch.from_numpy(training_x)
    var_y = torch.from_numpy(training_y)
    # print('y: ', var_y.shape)
    for episode in range(episodes):
        # forward
        out = model(var_x)
        # print(out.shape)
        loss = criterion(out, var_y)
        # print(loss)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if episode % 10 == 0:
            print('episode %d , Loss : %.5f' % (episode, loss.data))


def test_model(test_data_x, test_data_y, model):
    test_data_x = test_data_x.reshape(-1, 1, 32)
    # print('x: ', test_data_x)
    # print(training_data_y)
    test_data_y = np.array(test_data_y, dtype=np.float32)
    # print('y: ', training_y.shape)
    # print('yes')
    # print(training_data_y)
    test_data_y = test_data_y.reshape(-1, 1, 1)
    # print('y: ', training_y)
    var_x = torch.from_numpy(test_data_x)
    var_y = torch.from_numpy(test_data_y)
    num_wrong = 0
    num_test = len(test_data_y)
    ans = model(var_x)
    for index in range(len(ans)):
        if ans[index] > 1:
            ans[index] = 1
        if ans[index] < -1:
            ans[index] = -1
    # print(ans.shape)
    # print(var_y.shape)
    correct = abs(ans - var_y)
    for i in correct:
        if i > 0.5:
            num_wrong += 1
    print('accuracy : %.5f' % (1 - num_wrong / num_test))
