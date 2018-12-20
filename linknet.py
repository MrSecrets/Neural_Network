import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

nn.MaxPool2d(kernel_size, stride=None)
nn.ReLU()
nn.BatchNorm2d(num_features)
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1)

def encoder_block(input,m,n):
    layer_0 = nn.BatchNorm2d()(input)
    layer_1 = nn.Conv2D(in_channels=m, out_channels=n, kernel_size=(3,3), stride=2)(layer_1)
    layer_1 = nn.ReLU()(layer_0)

    layer_2 = nn.BatchNorm2d()(layer_1)
    layer_2 = nn.Conv2D(in_channels=n, out_channels=n, kernel_size=(3,3), stride=1)(layer_2)
    layer_2 = nn.ReLU()(layer_2)

    layer_added = torch.cat((layer_0,layer_2),0)

    layer_3 = nn.BatchNorm2d()(layer_added)
    layer_3 = nn.Conv2D(in_channels=n, out_channels=n, kernel_size=(3,3), stride=1)(layer_3)
    layer_3 = nn.ReLU()(layer_3)
    
    layer_4 = nn.BatchNorm2d()(layer_3)
    layer_4 = nn.Conv2D(in_channels=n, out_channels=n, kernel_size=(3,3), stride=1)(layer_4)
    layer_4 = nn.RelU()(layer_4)

    output_layer = torch.cat(layer_0,layer_4)

    return output_layer


def decoder_block(input,m,n):
    layer_1 = nn.BatchNorm2d()(input)
    layer_1 = nn.Conv2D(in_channels=m, out_channels=m/4, kernel_size=(1,1))(layer_1)
    layer_1 = nn.ReLU()(layer_1)

    layer_2 = nn.Upsample(scale_factor=2)(layer_2)
    layer_2 = nn.BatchNorm2d()(input)
    layer_2 = nn.Conv2D(in_channels=m/4, out_channels=m/4, kernel_size=(3,3))(layer_2)
    layer_2 = nn.ReLU()(layer_2)

    layer_3 = nn.BatchNorm2d()(layer_2)
    layer_3 = nn.Conv2D(in_channels=m/4, out_channels=n, kernel_size=(1,1))(layer_3)
    layer_3 = nn.ReLU()(layer_3)

    return layer_3


def initial_block(input):
    layer = nn.BatchNorm2d()(input)
    layer = nn.conv2D(in_channels=3, out_channels=64, kernel_size=(7,7), stride=2)(layer)
    layer = nn.MaxPool2d(kernel_size=(3,3), stride=2)(layer_1)(layer)
    layer = nn.ReLU()(layer)
    
    return layer

def final_block(input,N):
    layer_1 = nn.BatchNorm2d()(input)
    layer_1 = nn.Upsample(scale_factor=2)(layer_1)
    layer_1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=(3,3))(layer_1)
    layer_1 = nn.ReLU()(layer_1)

    layer_2 = nn.BatchNorm2d()(layer_1)
    layer_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(3,3))(layer_2)
    layer_2 = nn.ReLU()(layer_2)

    layer_3 = nn.BatchNorm2d()(layer_2)
    layer_3 = nn.Upsample(scale_factor=2)(layer_3)
    layer_3 = nn.Conv2D(in_channels=64, out_channels=N, kernel_size=(2,2))(layer_3)

    return layer_3



def linknet(inputs,n_classes):
    initial_layer = initial_block(inputs,3,64)
    
    encoder_1 = encoder_block(initial_layer,64,64)
    encoder_2 = encoder_block(encoder_1,64,128)
    encoder_3 = encoder_block(encoder_2,128,256)
    encoder_4 = encoder_block(encoder_3,256,512)
    
    decoder_4 = decoder_block(encoder_4,512,256)
    decoder_3 = decoder_block(torch.cat((encoder_3,decoder_4),0),256,128)
    decoder_2 = decoder_block(torch.cat((encoder_2,decoder_3),0),128,64)
    decoder_1 = decoder_block(torch.cat((encoder_1,decoder_2),0),64,64)

    model = final_block(decoder_1, n_classes)

    return model