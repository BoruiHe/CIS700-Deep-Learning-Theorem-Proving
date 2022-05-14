from unicodedata import name
from pip import main
import torch
from torch import nn


class cnn_1d_lstm(nn.Module):
    def __init__(self, voc_size):
        super(cnn_1d_lstm, self).__init__()
        self.embedding = nn.Embedding(voc_size, 256)
        self.conv1d_1 = nn.Conv1d(256, 256, 7)
        self.MaxPool1d_1 = nn.MaxPool1d(3, stride=3)
        self.conv1d_2 = nn.Conv1d(256, 256, 7)
        self.MaxPool1d_2 = nn.MaxPool1d(5, stride=5)
        self.lstm = nn.LSTM(256, 256, batch_first=True) 
        self.linear_256 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.dropout =nn.Dropout()
        self.linear_1 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0,2,1)
        x = self.conv1d_1(x)
        x = self.MaxPool1d_1(x)
        x = self.conv1d_2(x)
        x = self.MaxPool1d_2(x)
        x = x.permute(0,2,1)
        output, (h, c) = self.lstm(x)
        x = self.linear_256(h)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.sigmoid(x)
        return x
    
if __name__ == '__main__':
    model = cnn_1d_lstm(voc_size = 20)
    input = torch.randint(10, (2, 69,)).long()
    output = model(input)
    print('------------------------------------------------------------------------')
    print('Input size: {0}, Output size: {1}'.format(input.shape, output.shape))
pass