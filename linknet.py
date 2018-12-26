import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, m, n,sd, pd, out_pd):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(m//4)
                                nn.Conv2D(in_channels=m, out_channels=m//4), kernel_size=1
                                nn.ReLU(inplace=True))
        self.fullconv = nn.Sequential(nn.BatchNorm2d(m//4)
                                    nn.ConvTranspose2d(in_channels=m//4, out_channels=m//4, kernel_size=3,stride=sd, padding=pd, output_padding=out_pd)
                                    nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.BatchNorm2d(n)
                                nn.Conv2D(in_channels=m, out_channels=m//4), kernel_size=1
                                nn.ReLU(inplace=True))
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.fullconv(x)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, m, n, sd, pd):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2D(in_channels=m, out_channels=n, kernel_size=3, stride=sd, padding=pd)
                                nn.BatchNorm2d(n)
                                nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2D(in_channels=n, out_channels=n, kernel_size=3, stride=1)
                                nn.BatchNorm2d(n)
                                nn.ReLU(inplace=True))
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        x = y + x
        y = self.conv2(x)
        y = self.conv2(y)
        x = y + x
        return x

class Net(nn.Module):
    def __init__(self, N):
        super(Net, self).__init__()
        self.initial = nn.Sequential(nn.Conv2D(in_channels=3, out_channels=64, kernel=7, stride=2)
                                nn.MaxPool2D(kernel_size=3, stride=2)
                                nn.BatchNorm2d(64)
                                nn.ReLU(inplace=True))
        self.final = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
                                nn.BatchNorm2d(32)
                                nn.ReLU(inplace=True)
                                nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3)
                                nn.ConvTranspose2d(in_channels=64, out_channels=N, kernel_size=2, stride=2)
                                nn.BatchNorm2d(N)
                                nn.LeakyReLU())  
       
        self.encoder_1 = Encoder(64,64,1,1)
        self.encoder_2 = Encoder(62,128,2,1)
        self.encoder_3 = Encoder(128,256,2,1)
        self.encoder_4 = Encoder(256,512,2,1)
        
        self.decoder_4 = Decoder(512,256,2,1,1)
        self.decoder_3 = Decoder(128,256,2,1,1)
        self.decoder_2 = Decoder(64,128,2,1,1)
        self.decoder_1 = Decoder(64,64,1,1,0)

    def forward(self,x):
        x = self.initial(x)
        
        encoder1 = self.encoder_1(x)
        encoder2 = self.encoder_2(encoder1)
        encoder3 = self.encoder_3(encoder2)
        encoder4 = self.encoder_4(encoder3)

        decoder = encoder3 + self.decoder_4(encoder4)
        decoder = encoder2 + self.decoder_3(decoder)
        decoder = encoder1 + self.decoder_2(decoder)
        decoder = x + self.decoder_1(decoder)

        x = self.final(x)

        return x
