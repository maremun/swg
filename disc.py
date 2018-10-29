import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, C):
        super(Discriminator, self).__init__()
        
        self.C = C
        self.f_dim = 4*4*512

        self.l1 = nn.Sequential(nn.Conv2d(self.C, 64, 4, 2, 1),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU())
        
        self.l2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1),
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU())
        self.l3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1),
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU())
        self.l4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1),
                                nn.BatchNorm2d(512),
                                nn.LeakyReLU())
        self.l5 = nn.Sequential(nn.Linear(self.f_dim, 1, False))
        
        
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        features = out.view(x.size(0), -1)
        score = self.l5(features)
        return score, features
        