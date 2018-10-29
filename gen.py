import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, d):
        super(Generator, self).__init__()
        
        self.d = d
        self.l1 = nn.Sequential(nn.Linear(self.d, 4*4*1024),
                                nn.BatchNorm1d(4*4*1024),
                                nn.ReLU())
        
        self.l2 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1),
                                nn.BatchNorm2d(512),
                                nn.ReLU())
        
        self.l3 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU())
        
        self.l4 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU())
        
        self.l5 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU())
        
        self.l6 = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 2, bias=False),
                                nn.Sigmoid())
        
    def forward(self, z):
        x = self.l1(z)
        x = self.l2(x.view(-1, 1024, 4, 4))
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x