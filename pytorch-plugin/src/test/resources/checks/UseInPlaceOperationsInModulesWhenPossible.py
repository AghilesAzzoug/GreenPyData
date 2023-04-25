import torch.nn as nn
import torch


class NonCompliantReLU(nn.Module):
    def __init__(self):
        super(NonCompliantReLU, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1),
                        nn.MaxPool2d(2),
                        nn.ReLU(), # Noncompliant {{Use InPlace operations when possible.}}
                        nn.Conv2d(10, 20, kernel_size=5, bias=True),
                        nn.BatchNorm2d(20),
                        nn.MaxPool2d(2),
                        nn.ReLU(inplace=False) # Noncompliant {{Use InPlace operations when possible.}}
            )
        self.dense1 = nn.Linear(in_features=320, out_features=50)
        self.dense1_bn = nn.BatchNorm1d(50)
        self.dense2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 320)
        x = F.relu(self.dense1_bn(self.dense1(x)))
        return F.relu(self.dense2(x))

class NonCompliantHardtanh(nn.Module):
    def __init__(self):
        super(NonCompliantHardtanh, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1),
                        nn.MaxPool2d(2),
                        torch.nn.Hardtanh(), # Noncompliant {{Use InPlace operations when possible.}}
                        nn.Conv2d(10, 20, kernel_size=5, bias=True),
                        nn.BatchNorm2d(20),
                        nn.MaxPool2d(2),
                        torch.nn.Hardtanh(inplace=False) # Noncompliant {{Use InPlace operations when possible.}}
            )
        self.dense1 = nn.Linear(in_features=320, out_features=50)
        self.dense1_bn = nn.BatchNorm1d(50)
        self.dense2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 320)
        x = F.relu(self.dense1_bn(self.dense1(x)))
        return F.relu(self.dense2(x))

class NonCompliantLeakyRELU(nn.Module):
    def __init__(self):
        super(NonCompliantLeakyRELU, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1),
                        nn.MaxPool2d(2),
                        nn.LeakyReLU(), # Noncompliant {{Use InPlace operations when possible.}}
                        nn.Conv2d(10, 20, kernel_size=5, bias=True),
                        nn.BatchNorm2d(20),
                        nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(negative_slope=0.01, inplace=False) # Noncompliant {{Use InPlace operations when possible.}}
            )
        self.dense1 = nn.Linear(in_features=320, out_features=50)
        self.dense1_bn = nn.BatchNorm1d(50)
        self.dense2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 320)
        x = F.relu(self.dense1_bn(self.dense1(x)))
        return F.relu(self.dense2(x))

class CompliantReLU(nn.Module):
    def __init__(self):
        super(CompliantReLU, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1),
                        nn.MaxPool2d(2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(10, 20, kernel_size=5, bias=True),
                        nn.BatchNorm2d(20),
                        nn.MaxPool2d(2),
                        nn.ReLU(inplace=True)
            )
        self.dense1 = nn.Linear(in_features=320, out_features=50)
        self.dense1_bn = nn.BatchNorm1d(50)
        self.dense2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 320)
        x = F.relu(self.dense1_bn(self.dense1(x)))
        return F.relu(self.dense2(x))

class NonSequentialCompliantNet(nn.Module):
    def __init__(self):
        super(NonSequentialCompliantNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10,
                               kernel_size=5,
                               stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=False)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.dense1 = nn.Linear(in_features=320, out_features=50)
        self.dense1_bn = nn.BatchNorm1d(50)
        self.dense2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.dense1_bn(self.dense1(x)))
        return F.relu(self.dense2(x))
