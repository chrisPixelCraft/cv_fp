# model to predict 4 classifications
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

class CNNRNNModel(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, num_classes, rnn_layers=3, dropout=0.5):
        super(CNNRNNModel, self).__init__()

        # RNN layers with increased complexity
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers, batch_first=True, dropout=dropout)

        # Fully connected layers with increased complexity
        self.fc1 = nn.Linear(rnn_hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # Apply RNN layers
        rnn_out, _ = self.rnn(x)
        rnn_out_last = rnn_out[:, -1, :]

        # Apply fully connected layers
        x = self.fc1(rnn_out_last)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc3(x)
        return out
    

# class CNNModel(nn.Module):
#     def __init__(self, num_classes, dropout=0.5):
#         super(CNNModel, self).__init__()

#         # resnet as CNN layers
#         self.resnet18 = models.resnet18(pretrained=True)

#         # Fully connected layers with increased complexity
# #         self.fc = nn.Sequential(
# #             nn.Linear(512, 256),
# #             nn.Linear(256, 128),
# #             nn.Linear(128, num_classes),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #         )
#         self.resnet18.fc = nn.Identity()
#         self.fc1 = nn.Linear(512, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#         # Initialize weights
# #         self._init_weights()

#     def _init_weights(self):
#         init.xavier_uniform_(self.fc1.weight)
#         init.xavier_uniform_(self.fc2.weight)
#         init.xavier_uniform_(self.fc3.weight)

#     def forward(self, x):
#         x = self.resnet18(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         out = self.fc3(x)
        
#         return out


if __name__ == "__main__":
    input_size = 2048  # Example input size, should match your precomputed feature size
    model = CNNRNNModel(num_classes=4, dropout=0.5)
    print(model)


## model with softmax
# import torch
# import torch.nn as nn
# import torch.nn.init as init

# class CNNRNNModel(nn.Module):
#     def __init__(self, input_size, rnn_hidden_size, num_classes, rnn_layers=3, dropout=0.5):
#         super(CNNRNNModel, self).__init__()

#         # RNN layers with increased complexity
#         self.rnn = nn.LSTM(input_size=input_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers, batch_first=True, dropout=dropout)

#         # Fully connected layers with increased complexity
#         self.fc1 = nn.Linear(rnn_hidden_size, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)  # Add softmax layer

#         # Initialize weights
#         self._init_weights()

#     def _init_weights(self):
#         init.xavier_uniform_(self.fc1.weight)
#         init.xavier_uniform_(self.fc2.weight)
#         init.xavier_uniform_(self.fc3.weight)

#     def forward(self, x):
#         # Apply RNN layers
#         rnn_out, _ = self.rnn(x)
#         # Apply fully connected layers to each time step
#         x = self.fc1(rnn_out)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         out = self.fc3(x)
#         out = self.softmax(out)  # Apply softmax activation
#         return out

# if __name__ == "__main__":
#     input_size = 2048  # Example input size, should match your precomputed feature size
#     model = CNNRNNModel(input_size=input_size, rnn_hidden_size=512, num_classes=4, rnn_layers=3, dropout=0.5)
#     print(model)




## model strong enough
# import torch
# import torch.nn as nn
# import torch.nn.init as init

# class CNNRNNModel(nn.Module):
#     def __init__(self, input_size, rnn_hidden_size, num_classes, rnn_layers=1, dropout=0.5):
#         super(CNNRNNModel, self).__init__()

#         # RNN layers
#         self.rnn = nn.LSTM(input_size=input_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers, batch_first=True, dropout=dropout)

#         # Fully connected layers
#         self.fc1 = nn.Linear(rnn_hidden_size, 128)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(128, num_classes)

#         # Initialize weights
#         self._init_weights()

#     def _init_weights(self):
#         init.xavier_uniform_(self.fc1.weight)
#         init.xavier_uniform_(self.fc2.weight)

#     def forward(self, x):
#         # Apply RNN layers
#         rnn_out, _ = self.rnn(x)
#         rnn_out_last = rnn_out[:, -1, :]

#         # Apply fully connected layers
#         x = self.fc1(rnn_out_last)
#         x = self.relu(x)
#         x = self.dropout(x)
#         out = self.fc2(x)
#         return out

# if __name__ == "__main__":
#     input_size = 2048  # Example input size, should match your precomputed feature size
#     model = CNNRNNModel(input_size=input_size, rnn_hidden_size=512, num_classes=4, rnn_layers=2, dropout=0.5)
#     print(model)



## conv1d with maxpool
# import torch
# import torch.nn as nn
# import torch.nn.init as init

# class CNNRNNModel(nn.Module):
#     def __init__(self, input_size, rnn_hidden_size, num_classes, rnn_layers=1, dropout=0.5):
#         super(CNNRNNModel, self).__init__()

#         # CNN layers
#         self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(64)
#         # self.pool = nn.MaxPool1d(kernel_size=2)
#         print("pool")
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(128)

#         # RNN layers
#         self.rnn = nn.LSTM(input_size=128, hidden_size=rnn_hidden_size, num_layers=rnn_layers, batch_first=True, dropout=dropout)

#         # Fully connected layers
#         self.fc1 = nn.Linear(rnn_hidden_size, 128)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(128, num_classes)

#         # Initialize weights
#         self._init_weights()

#     def _init_weights(self):
#         init.xavier_uniform_(self.fc1.weight)
#         init.xavier_uniform_(self.fc2.weight)
#         init.xavier_uniform_(self.conv1.weight)
#         init.xavier_uniform_(self.conv2.weight)

#     def forward(self, x):
#         # Apply CNN layers
#         x = x.permute(0, 2, 1)  # Change to (batch_size, input_size, sequence_length)
#         x = self.pool(self.relu(self.bn1(self.conv1(x))))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = x.permute(0, 2, 1)  # Change back to (batch_size, sequence_length, features)

#         # Apply RNN layers
#         rnn_out, _ = self.rnn(x)
#         rnn_out_last = rnn_out[:, -1, :]

#         # Apply fully connected layers
#         x = self.fc1(rnn_out_last)
#         x = self.relu(x)
#         x = self.dropout(x)
#         out = self.fc2(x)
#         return out

# if __name__ == "__main__":
#     input_size = 2048  # Example input size
#     model = CNNRNNModel(input_size=input_size, rnn_hidden_size=512, num_classes=4, rnn_layers=2, dropout=0.5)
#     print(model)



## basic model
# import torch
# import torch.nn as nn

# class CNNRNNModel(nn.Module):
#     def __init__(self, input_size, rnn_hidden_size, num_classes, rnn_layers=1):
#         super(CNNRNNModel, self).__init__()
#         self.rnn = nn.LSTM(input_size=input_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers, batch_first=True)
#         self.fc = nn.Linear(rnn_hidden_size, num_classes)

#     def forward(self, x):
#         rnn_out, _ = self.rnn(x)
#         rnn_out_last = rnn_out[:, -1, :]
#         out = self.fc(rnn_out_last)
#         return out

# if __name__ == "__main__":
#     input_size = 2048  # Example input size
#     model = CNNRNNModel(input_size=input_size, rnn_hidden_size=512, num_classes=4)
#     print(model)
