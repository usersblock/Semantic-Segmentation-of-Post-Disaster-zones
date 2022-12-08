import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vgg13_bn, vgg16_bn, vgg16, vgg19, vgg19_bn
from torchvision.models.resnet import resnet50
#import albumentations as album

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,bn = False):
        super(DoubleConv, self).__init__()
        if(bn):
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)
    
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels,bn = False):
        super(DownBlock, self).__init__()
        if(bn):
            self.double_conv = DoubleConv(in_channels, out_channels,True)
        else:
            self.double_conv = DoubleConv(in_channels, out_channels,False)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode,bn = False):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        if(bn):
            self.double_conv = DoubleConv(in_channels, out_channels,True)
        else:
            self.double_conv = DoubleConv(in_channels, out_channels,False)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class UpBlockMid(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, up_sample_mode,bn = False):
        super(UpBlockMid, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        if(bn):
            self.double_conv = DoubleConv(mid_channels, out_channels,True)
        else:
            self.double_conv = DoubleConv(mid_channels, out_channels,False)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)    

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)    
    
class UNet(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', batch_norm=False):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        self.bn = batch_norm
        # Downsampling Path
        self.down_conv1 = DownBlock(3, 64,self.bn)
        self.down_conv2 = DownBlock(64, 128,self.bn)
        self.down_conv3 = DownBlock(128, 256,self.bn)
        self.down_conv4 = DownBlock(256, 512,self.bn)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024,self.bn)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode,self.bn)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode,self.bn)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode,self.bn)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode,self.bn)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x
class resnet_unet(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(resnet_unet, self).__init__()
        
        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([128, 192, 256, 512, 1024]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        
        self.res = nn.Conv2d(decoder_filters[-5], 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = resnet50(pretrained=pretrained)

        self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4


    def forward(self, x):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class resnet_siamunet(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(resnet_siamunet, self).__init__()

        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        
        self.res = nn.Conv2d(decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = resnet50(pretrained=pretrained)

        self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4


    def forward1(self, x):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10


    def forward(self, x1, x2):

        dec10_0 = self.forward1(x1)
        dec10_1 = self.forward1(x2)

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class VGG16Unet(nn.Module):


    def __init__(self,out_channels=2):
        super().__init__()

        self.encoder = vgg16(pretrained=True).features
        self.block1 = nn.Sequential(*self.encoder[:4])
        self.block2 = nn.Sequential(*self.encoder[4:9])
        self.block3 = nn.Sequential(*self.encoder[9:16])
        self.block4 = nn.Sequential(*self.encoder[16:23])
        self.block5 = nn.Sequential(*self.encoder[23:30])

        self.bottleneck = nn.Sequential(*self.encoder[30:])
        self.conv_bottleneck = double_conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = double_conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = double_conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x
                
class SiamUNet(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', batch_norm=False):
        super(SiamUNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        self.bn = batch_norm
        # Downsampling Path
        self.down_conv1 = DownBlock(6, 64,self.bn)
        self.down_conv2 = DownBlock(64, 128,self.bn)
        self.down_conv3 = DownBlock(128, 256,self.bn)
        self.down_conv4 = DownBlock(256, 512,self.bn)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024,self.bn)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode,self.bn)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode,self.bn)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode,self.bn)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode,self.bn)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x1,x2):
        xconcat = torch.cat((x1,x2),dim=1)
        x, skip1_out = self.down_conv1(xconcat)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x
    
class SiamUNetConC(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose',batch_norm=False):
        super(SiamUNetConC, self).__init__()
        self.up_sample_mode = up_sample_mode
        self.bn = batch_norm
        # Downsampling Path
        self.down_conv1 = DownBlock(3, 64,self.bn)
        self.down_conv2 = DownBlock(64, 128,self.bn)
        self.down_conv3 = DownBlock(128, 256,self.bn)
        self.down_conv4 = DownBlock(256, 512,self.bn)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024,self.bn)
        # Upsampling Path
        self.up_conv4 = UpBlockMid(1024,2048,512, self.up_sample_mode,self.bn)
        self.up_conv3 = UpBlockMid(512,1024,256,self.up_sample_mode,self.bn)
        self.up_conv2 = UpBlockMid(256,512,128,self.up_sample_mode,self.bn)
        self.up_conv1 = UpBlockMid(128,256,64,self.up_sample_mode,self.bn)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x1,x2):
        #input A
        x1, skip11_out = self.down_conv1(x1)
        x1, skip21_out = self.down_conv2(x1)
        x1, skip31_out = self.down_conv3(x1)
        x1, skip41_out = self.down_conv4(x1)
        
        #input B
        x2, skip12_out = self.down_conv1(x2)
        x2, skip22_out = self.down_conv2(x2)
        x2, skip32_out = self.down_conv3(x2)
        x2, skip42_out = self.down_conv4(x2)
        
        x = self.double_conv(x2)
        x = self.up_conv4(x, torch.cat((skip41_out,skip42_out),dim = 1))
        x = self.up_conv3(x, torch.cat((skip31_out,skip32_out),dim = 1))
        x = self.up_conv2(x, torch.cat((skip21_out,skip22_out),dim = 1))
        x = self.up_conv1(x, torch.cat((skip11_out,skip12_out),dim = 1))
        x = self.conv_last(x)
        return x
    
class SiamUNetDiff(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose',batch_norm = False):
        super(SiamUNetDiff, self).__init__()
        self.up_sample_mode = up_sample_mode
        self.bn = batch_norm
        # Downsampling Path
        self.down_conv1 = DownBlock(3, 64,self.bn)
        self.down_conv2 = DownBlock(64, 128,self.bn)
        self.down_conv3 = DownBlock(128, 256,self.bn)
        self.down_conv4 = DownBlock(256, 512,self.bn)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode,self.bn)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode,self.bn)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode,self.bn)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode,self.bn)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x1,x2):
        #input A
        x1, skip11_out = self.down_conv1(x1)
        x1, skip21_out = self.down_conv2(x1)
        x1, skip31_out = self.down_conv3(x1)
        x1, skip41_out = self.down_conv4(x1)
        
        #input B
        x2, skip12_out = self.down_conv1(x2)
        x2, skip22_out = self.down_conv2(x2)
        x2, skip32_out = self.down_conv3(x2)
        x2, skip42_out = self.down_conv4(x2)
        
        x = self.double_conv(x2)
        x = self.up_conv4(x, torch.abs(skip41_out - skip42_out))
        x = self.up_conv3(x, torch.abs(skip31_out - skip32_out))
        x = self.up_conv2(x, torch.abs(skip21_out - skip22_out))
        x = self.up_conv1(x, torch.abs(skip11_out - skip12_out))
        x = self.conv_last(x)
        return x

def double_conv(in_channels, out_channels,bn = True):
    if(bn == True):
        layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    else:
        layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True))
        
    return layers


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


class VGGUnet(nn.Module):
    """Unet with VGG-13 (with BN), VGG-16 (with BN) encoder.
    """

    def __init__(self, encoder, *, pretrained=False, out_channels=2):
        super().__init__()

        self.encoder = encoder(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = double_conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = double_conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = double_conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x

class VGGUnet19(nn.Module):
    """Unet with VGG-13 (with BN), VGG-16 (with BN) encoder.
    """

    def __init__(self, encoder, *, pretrained=False, out_channels=2):
        super().__init__()

        self.encoder = encoder(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:26])
        self.block4 = nn.Sequential(*self.encoder[26:39])
        self.block5 = nn.Sequential(*self.encoder[39:52])

        self.bottleneck = nn.Sequential(*self.encoder[52:])
        self.conv_bottleneck = double_conv(512, 1024,True)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = double_conv(512 + 512, 512,True)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256,True)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128,True)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 128, 64,True)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = double_conv(32 + 64, 32,True)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x    
    
class VGGUnet19nobn(nn.Module):
    """Unet with VGG-13 (with BN), VGG-16 (with BN) encoder.
    """

    def __init__(self, encoder, *, pretrained=True, out_channels=5):
        super().__init__()

        self.conv_bottleneck = double_conv(512, 1024,False)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = double_conv(512 + 512, 512, False)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256, False)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128, False)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 128, 64, False)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = double_conv(32 + 64, 32, False)
        self.conv12 = nn.Conv2d(32, out_channels, kernel_size=1)

        #self._initialize_weights()
        
        self.encoder = encoder(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:4])
        self.block2 = nn.Sequential(*self.encoder[4:9])
        self.block3 = nn.Sequential(*self.encoder[9:18])
        self.block4 = nn.Sequential(*self.encoder[18:27])
        self.block5 = nn.Sequential(*self.encoder[27:36])

        self.bottleneck = nn.Sequential(*self.encoder[36:])        
        
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv12(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    
class VGGUnet19nobndouble(nn.Module):
    """Unet with VGG-13 (with BN), VGG-16 (with BN) encoder.
    """

    def __init__(self, encoder, *, pretrained=False, out_channels=2):
        super().__init__()

        self.encoder = encoder(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:4])
        self.block2 = nn.Sequential(*self.encoder[4:9])
        self.block3 = nn.Sequential(*self.encoder[9:18])
        self.block4 = nn.Sequential(*self.encoder[18:27])
        self.block5 = nn.Sequential(*self.encoder[27:36])

        self.bottleneck = nn.Sequential(*self.encoder[36:])
        self.conv_bottleneck = double_conv(512, 1024,False)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = double_conv(512 + 512, 512, False)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256, False)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128, False)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 128, 64, False)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = double_conv(32 + 64, 32, False)
        self.conv12 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward1(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)
        
        return x
    def forward(self,x1,x2):
        x1 = self.forward1(x1)
        x2 = self.forward1(x2)
        xconcat = torch.cat([x1,x2],1)
        x = self.conv12(xconcat)
        return x      
    
class SiamUNetDiffVgg19(nn.Module):
    def __init__(self,out_channels=5):
        super().__init__()
        self.encoder = vgg19(pretrained=True).features
        self.block1 = nn.Sequential(*self.encoder[:4])
        self.block2 = nn.Sequential(*self.encoder[4:9])
        self.block3 = nn.Sequential(*self.encoder[9:18])
        self.block4 = nn.Sequential(*self.encoder[18:27])
        self.block5 = nn.Sequential(*self.encoder[27:36])

        self.bottleneck = nn.Sequential(*self.encoder[36:])
        self.conv_bottleneck = double_conv(512, 1024,False)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = double_conv(512 + 512, 512, False)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256, False)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128, False)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 128, 64, False)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = double_conv(32 + 64, 32, False)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        #x1 branch
        block11 = self.block1(x1)
        block12 = self.block2(block11)
        block13 = self.block3(block12)
        block14 = self.block4(block13)
        block15 = self.block5(block14)
        #x2 branch
        block21 = self.block1(x2)
        block22 = self.block2(block21)
        block23 = self.block3(block22)
        block24 = self.block4(block23)
        block25 = self.block5(block24)

        bottleneck = self.bottleneck(block25)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, torch.abs(block15-block25)], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, torch.abs(block14-block24)], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, torch.abs(block13-block23)], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, torch.abs(block12-block22)], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, torch.abs(block11-block11)], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x   
        
class SiamUNetDiffVgg19Space(nn.Module):
    def __init__(self,out_channels=5):
        super().__init__()
        self.encoder = vgg19space(num_classes = 5,pretrained=True,progress = True).features
        self.block1 = nn.Sequential(*self.encoder[:4])
        self.block2 = nn.Sequential(*self.encoder[4:9])
        self.block3 = nn.Sequential(*self.encoder[9:18])
        self.block4 = nn.Sequential(*self.encoder[18:27])
        self.block5 = nn.Sequential(*self.encoder[27:36])

        self.bottleneck = nn.Sequential(*self.encoder[36:])
        self.conv_bottleneck = double_conv(512, 1024,False)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = double_conv(512 + 512, 512, False)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256, False)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128, False)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 128, 64, False)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = double_conv(32 + 64, 32, False)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        #x1 branch
        block11 = self.block1(x1)
        block12 = self.block2(block11)
        block13 = self.block3(block12)
        block14 = self.block4(block13)
        block15 = self.block5(block14)
        #x2 branch
        block21 = self.block1(x2)
        block22 = self.block2(block21)
        block23 = self.block3(block22)
        block24 = self.block4(block23)
        block25 = self.block5(block24)

        blockdiff = torch.abs(block15-block25)
        bottleneck = self.bottleneck(blockdiff)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, torch.abs(block15-block25)], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, torch.abs(block14-block24)], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, torch.abs(block13-block23)], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, torch.abs(block12-block22)], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, torch.abs(block11-block11)], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x       
    
class SiamUNetConCVgg19(nn.Module):
    """Unet with VGG-13 (with BN), VGG-16 (with BN) encoder.
    """

    def __init__(self,out_channels=5):
        super().__init__()
        self.bn = False
        self.up_sample_mode = 'conv_transpose'
        self.encoder = vgg19(pretrained=True).features
        self.block1 = nn.Sequential(*self.encoder[:4])
        self.block2 = nn.Sequential(*self.encoder[4:9])
        self.block3 = nn.Sequential(*self.encoder[9:18])
        self.block4 = nn.Sequential(*self.encoder[18:27])
        self.block5 = nn.Sequential(*self.encoder[27:36])

        self.bottleneck = nn.Sequential(*self.encoder[36:])
        # Upsampling Path
        self.up_conv5 = UpBlockMid(512,1536,512, self.up_sample_mode,self.bn)
        self.up_conv4 = UpBlockMid(512,1536,256,self.up_sample_mode,self.bn)
        self.up_conv3 = UpBlockMid(256,768,128,self.up_sample_mode,self.bn)
        self.up_conv2 = UpBlockMid(128,384,64,self.up_sample_mode,self.bn)
        self.up_conv1 = UpBlockMid(64,192,64,self.up_sample_mode,self.bn)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x1,x2):
        #input A
        x11 = self.block1(x1)
        x12 = self.block2(x11)
        x13 = self.block3(x12)
        x14 = self.block4(x13)
        x15 = self.block5(x14)
        #input B
        x21 = self.block1(x2)
        x22 = self.block2(x21)
        x23 = self.block3(x22)
        x24 = self.block4(x23)
        x25 = self.block5(x24)
        
        
        x = self.bottleneck(x25)
        x = self.up_conv5(x, torch.cat((x15,x25),dim = 1))
        x = self.up_conv4(x, torch.cat((x14,x24),dim = 1))
        x = self.up_conv3(x, torch.cat((x13,x23),dim = 1))
        x = self.up_conv2(x, torch.cat((x12,x22),dim = 1))
        x = self.up_conv1(x, torch.cat((x11,x21),dim = 1))
        x = self.conv_last(x)
        return x
class SiamUNetFullConCVgg19(nn.Module):
    """Unet with VGG-13 (with BN), VGG-16 (with BN) encoder.
    """

    def __init__(self,out_channels=5):
        super().__init__()

        self.encoder = vgg19(pretrained=True).features
        self.block1 = nn.Sequential(*self.encoder[:4])
        self.block2 = nn.Sequential(*self.encoder[4:9])
        self.block3 = nn.Sequential(*self.encoder[9:18])
        self.block4 = nn.Sequential(*self.encoder[18:27])
        self.block5 = nn.Sequential(*self.encoder[27:36])

        self.bottleneck = nn.Sequential(*self.encoder[36:])
        self.conv_bottleneck = double_conv(512, 1024,False)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = double_conv(512 + 1024, 512, False)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 1024, 256, False)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 512, 128, False)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 256, 64, False)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = double_conv(32 + 128, 32, False)
        self.conv11 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        #x1 branch
        block11 = self.block1(x1)
        block12 = self.block2(block11)
        block13 = self.block3(block12)
        block14 = self.block4(block13)
        block15 = self.block5(block14)
        #x2 branch
        block21 = self.block1(x2)
        block22 = self.block2(block21)
        block23 = self.block3(block22)
        block24 = self.block4(block23)
        block25 = self.block5(block24)

        bottleneck2 = self.bottleneck(block25)
        bottleneck1 = self.bottleneck(block15)
        
        x1 = self.conv_bottleneck(bottleneck1)

        x1 = self.up_conv6(x1)
        x1 = torch.cat([x1, torch.cat([block15,block25], dim=1)], dim=1)
        x1 = self.conv6(x1)

        x1 = self.up_conv7(x1)
        x1 = torch.cat([x1, torch.cat([block14,block24], dim=1)], dim=1)
        x1 = self.conv7(x1)

        x1 = self.up_conv8(x1)
        x1 = torch.cat([x1, torch.cat([block13,block23], dim=1)], dim=1)
        x1 = self.conv8(x1)

        x1 = self.up_conv9(x1)
        x1 = torch.cat([x1, torch.cat([block12,block22], dim=1)], dim=1)
        x1 = self.conv9(x1)

        x1 = self.up_conv10(x1)
        x1 = torch.cat([x1, torch.cat([block11,block21], dim=1)], dim=1)
        x1 = self.conv10(x1)
        
        x2 = self.conv_bottleneck(bottleneck2)

        x2 = self.up_conv6(x2)
        x2 = torch.cat([x2, torch.cat([block15,block25], dim=1)], dim=1)
        x2 = self.conv6(x2)

        x2 = self.up_conv7(x2)
        x2 = torch.cat([x2, torch.cat([block14,block24], dim=1)], dim=1)
        x2 = self.conv7(x2)

        x2 = self.up_conv8(x2)
        x2 = torch.cat([x2, torch.cat([block13,block23], dim=1)], dim=1)
        x2 = self.conv8(x2)

        x2 = self.up_conv9(x2)
        x2 = torch.cat([x2, torch.cat([block12,block22], dim=1)], dim=1)
        x2 = self.conv9(x2)

        x2 = self.up_conv10(x2)
        x2 = torch.cat([x2, torch.cat([block11,block21], dim=1)], dim=1)
        x2 = self.conv10(x2)
        
        xconcat = torch.cat([x1,x2], dim = 1)
        
        x = self.conv11(xconcat)

        return x           

class UNETResnet50(nn.Module):
    """Deep Unet with ResNet50, ResNet101 or ResNet152 encoder.
    """

    def __init__(self, pretrained=False, out_channels=2):
        super().__init__()
        self.encoder = resnet50(pretrained=pretrained)
        self.encoder_layers = list(self.encoder.children())

        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        self.up_conv6 = up_conv(2048, 512)
        self.conv6 = double_conv(512 + 1024, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 64, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        x = self.up_conv6(block5)
        x = torch.cat([x, block4], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = self.conv10(x)

        return x 
    

class SiamUNetConCResnet50(nn.Module):
    """Deep Unet with ResNet50, ResNet101 or ResNet152 encoder.
    """

    def __init__(self, pretrained=False, out_channels=2):
        super().__init__()
        self.encoder = resnet50(pretrained=pretrained)
        self.encoder_layers = list(self.encoder.children())

        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        self.up_conv6 = up_conv(4096, 1024)
        self.conv6 = double_conv(1024 + 2048, 1024)
        self.up_conv7 = up_conv(1024, 512)
        self.conv7 = double_conv(512 + 1024, 512)
        self.up_conv8 = up_conv(512, 256)
        self.conv8 = double_conv(256 + 512, 256)
        self.up_conv9 = up_conv(256, 128)
        self.conv9 = double_conv(128 + 128, 128)
        self.up_conv10 = up_conv(128, 64)
        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x1,x2):
        block11 = self.block1(x1)
        block12 = self.block2(block11)
        block13 = self.block3(block12)
        block14 = self.block4(block13)
        block15 = self.block5(block14)

        block21 = self.block1(x2)
        block22 = self.block2(block21)
        block23 = self.block3(block22)
        block24 = self.block4(block23)
        block25 = self.block5(block24)
        xconcat = torch.cat([block15,block25],1)
        x = self.up_conv6(xconcat)
        x = torch.cat([x, torch.cat([block14,block24], dim=1)], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, torch.cat([block13,block23], dim=1)], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, torch.cat([block12,block22], dim=1)], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, torch.cat([block11,block21], dim=1)], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = self.conv10(x)

        return x 
    
    
def vgg13bn_unet(output_dim: int=2, pretrained: bool=False):
    return VGGUnet(vgg13_bn, pretrained=pretrained, out_channels=output_dim)


def vgg16bn_unet(output_dim: int=2, pretrained: bool=False):
    return VGGUnet(vgg16_bn, pretrained=pretrained, out_channels=output_dim)

def vgg19bn_unet(output_dim: int=2, pretrained: bool=False):
    return VGGUnet19(vgg19_bn, pretrained=pretrained, out_channels=output_dim)

def vgg19nobn_unet(output_dim: int=2, pretrained: bool=False):
    return VGGUnet19nobn(vgg19, pretrained=pretrained, out_channels=output_dim)
def vgg19nobn_unetdouble(output_dim: int=2, pretrained: bool=False):
    return VGGUnet19nobndouble(vgg19, pretrained=pretrained, out_channels=output_dim)

    
class VGG19Unet(nn.Module):
    """Unet with VGG-13 (with BN), VGG-16 (with BN) encoder.
    """

    def __init__(self, encoder, *, pretrained=False, out_channels=2):
        super().__init__()

        self.encoder = encoder(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = double_conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = double_conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = double_conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

class SiamUnet_diff_Full_Vgg19bn(nn.Module):
    """SiamUnet_diff segmentation network."""

    def __init__(self, input_nbr, label_nbr):
        super(SiamUnet_diff_Full_Vgg19bn, self).__init__()

        n1 = 64
        print(n1)
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.input_nbr = input_nbr

        self.conv11 = nn.Conv2d(input_nbr, filters[0], kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(filters[0])
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(filters[0])
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(filters[1])
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(filters[1])
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(filters[2])
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(filters[2])
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(filters[2])
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(filters[3])
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(filters[3])
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(filters[3])
        self.do43 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(filters[3])
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(filters[3])
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(filters[2])
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(filters[2])
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(filters[2])
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(filters[1])
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(filters[1])
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(filters[0])
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(filters[0])
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(filters[0], label_nbr, kernel_size=3, padding=1)

        self.sm = nn.LogSoftmax(dim=1)
        

    def forward(self, x1, x2):


        """Forward method."""
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)

        ####################################################
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)



        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return self.sm(x11d)