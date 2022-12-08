""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch
import torch.nn as nn
import torchvision
resnet = torchvision.models.resnet.resnet50(pretrained=True)
vgg = torchvision.models.vgg19_bn(weights='IMAGENET1K_V1')


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


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=5,pretrained = 'pretrained',load = ''):
        super().__init__()
        if(pretrained=='pretrained'):
            resnet = torchvision.models.resnet.resnet50(pretrained=True)
        elif(pretrained=='custom'):
            resnet = torchvision.models.resnet.resnet50(pretrained=False)
            resnet.load_state_dict(torch.load(load, map_location='cuda'))
            print('loaded successfully')
        else:
            resnet = torchvision.models.resnet.resnet50(pretrained=False)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            print(key)
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

class UNetWithVgg19BnEncoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=5):
        super().__init__()
        vgg = torchvision.models.vgg19_bn(weights='IMAGENET1K_V1')
        down_blocks = []
        up_blocks = []
        down_blocks.append(nn.Sequential(*list(vgg.children()))[0][0:7])
        down_blocks.append(nn.Sequential(*list(vgg.children()))[0][7:14])
        down_blocks.append(nn.Sequential(*list(vgg.children()))[0][14:27])
        down_blocks.append(nn.Sequential(*list(vgg.children()))[0][27:40])
        down_blocks.append(nn.Sequential(*list(vgg.children()))[0][40:53])
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(512, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()

        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i == (UNetWithVgg19BnEncoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x
        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithVgg19BnEncoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x                

class SiameseUNetV2(nn.Module):
    DEPTH = 6

    def __init__(self, n_channels = 3,n_classes=5):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        
        down_blocks = []
        up_blocks = []
        down_blocks.append(Down(64, 128))
        down_blocks.append(Down(128, 256))
        down_blocks.append(Down(256, 512))
        down_blocks.append(Down(512, 1024))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 1024)
        up_blocks.append(SiameseUp(1024, 3072, 512,True))
        up_blocks.append(SiameseUp(512, 1536, 256,True))
        up_blocks.append(SiameseUp(256, 768, 128,True))
        up_blocks.append(SiameseUp(128, 384, 64,True))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = OutConv(64, n_classes)

    def forward(self, x1, x2, with_output_feature_map=False):
        x1 = self.inc(x1)
        x2 = self.inc(x2)
        
        x11 = self.down_blocks[0](x1)
        x12 = self.down_blocks[0](x2)
        x21 = self.down_blocks[1](x11)
        x22 = self.down_blocks[1](x12)
        x31 = self.down_blocks[2](x21)
        x32 = self.down_blocks[2](x22)
        x41 = self.down_blocks[3](x31)
        x42 = self.down_blocks[3](x32)
        xconcat = torch.cat((x41,x42),dim = 1)
        xb = self.bridge(xconcat)
        xu1 = self.up_blocks[0](x42,torch.cat((x41,x42),dim = 1))
        xu2 = self.up_blocks[1](xu1,torch.cat((x31,x32),dim = 1))
        xu3 = self.up_blocks[2](xu2,torch.cat((x21,x22),dim = 1))
        xu4 = self.up_blocks[3](xu3,torch.cat((x11,x12),dim = 1))

        x = self.out(xu4)
        return x

'''class SiameseUNetV2ConcBegin(nn.Module):
    DEPTH = 6

    def __init__(self, n_channels = 3,n_classes=5):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        
        down_blocks = []
        up_blocks = []
        down_blocks.append(Down(64, 128))
        down_blocks.append(Down(128, 256))
        down_blocks.append(Down(256, 512))
        down_blocks.append(Down(512, 1024))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 1024)
        up_blocks.append(SiameseUp(1024, 3072, 512,True))
        up_blocks.append(SiameseUp(512, 1536, 256,True))
        up_blocks.append(SiameseUp(256, 768, 128,True))
        up_blocks.append(SiameseUp(128, 384, 64,True))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = OutConv(64, n_classes)

    def forward(self, x1, x2, with_output_feature_map=False):
        x = self.inc(torch.cat((x1,x2),dim=1))
        
        x1 = self.down_blocks[0](x)
        x2 = self.down_blocks[1](x1)
        x3 = self.down_blocks[2](x2)
        x4 = self.down_blocks[3](x3)
        xu1 = self.up_blocks[0](x4,,dim = 1)
        xu2 = self.up_blocks[1](xu1,,dim = 1)
        xu3 = self.up_blocks[2](xu2,torch.cat((x21,x22),dim = 1))
        xu4 = self.up_blocks[3](xu3,torch.cat((x11,x12),dim = 1))

        x = self.out(xu4)
        return x    
'''    
class SiameseUNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=5):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(1024, 1024)        
        up_blocks.append(SiameseUp(2048, 6144, 1024,False))
        up_blocks.append(SiameseUp(1024, 512 + (1024*2), 512,False))
        up_blocks.append(SiameseUp(512, 256 + (512*2), 256,False))
        up_blocks.append(SiameseUp(256, 128 + (256*2), 128,False))
        up_blocks.append(SiameseUp(128, 64 + (64*2), 64,False))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = OutConv(64, n_classes)

    def forward(self, x1, x2):
        dblock = []
        for i, block in enumerate(self.down_blocks, 2):
            dblock.append(block)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
        x11 = self.input_block(x1)
        x21 = self.input_block(x2)
        xp12 = self.input_pool(x11)
        xp22 = self.input_pool(x21)
        x12 = dblock[0](xp12)
        x22 = dblock[0](xp22)
        x13 = dblock[1](x12)
        x23 = dblock[1](x22)
        x14 = dblock[2](x13)
        x24 = dblock[2](x23)
        #x15 = dblock[3](x14)
        #x25 = dblock[3](x24)
        xbridge = self.bridge(x24)
        #xu1 = self.up_blocks[0](xbridge,torch.cat((x15,x25),dim = 1))
        xu2 = self.up_blocks[1](xbridge,torch.cat((x14,x24),dim = 1))
        xu3 = self.up_blocks[2](xu2,torch.cat((x13,x23),dim = 1))
        xu4 = self.up_blocks[3](xu3,torch.cat((x12,x22),dim = 1))
        xu5 = self.up_blocks[4](xu4,torch.cat((x11,x21),dim = 1))
        x = self.out(xu5)
        return x
        '''pre_pools1 = dict()
        pre_pools2 = dict()
        pre_pools1[f"layer_0"] = x1
        pre_pools2[f"layer_0"] = x2
        x1 = self.input_block(x1)
        x2 = self.input_block(x2)
        pre_pools1[f"layer_1"] = x1
        pre_pools2[f"layer_1"] = x2
        x1 = self.input_pool(x1)
        x2 = self.input_pool(x2)

        for i, block in enumerate(self.down_blocks, 2):
            x1 = block(x1)
            x2 = block(x2)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x1
            pre_pools[f"layer_{i}"] = x2
        '''



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class SiameseUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SiameseUNet, self).__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(*list(resnet.children()))[:3]
        #self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.outc = OutConv(128, n_classes)

    def forward(self, xa, xb):
        # Input A
        xa1 = self.inc(xa)
        xa2 = self.down1(xa1)
        xa3 = self.down2(xa2)
        xa4 = self.down3(xa3)
        xa5 = self.down4(xa4)
        
        # Input B
        xb1 = self.inc(xb)
        xb2 = self.down1(xb1)
        xb3 = self.down2(xb2)
        xb4 = self.down3(xb3)
        xb5 = self.down4(xb4)
 
        xconcat = torch.cat((xa5,xb5),dim = 1)
        xcat4 = torch.cat((xa4,xb4),dim = 1)
        x = self.up1(xconcat, xcat4)
        xcat3 = torch.cat((xa3,xb3),dim = 1)
        x = self.up2(x, xcat3)
        xcat2 = torch.cat((xa2,xb2),dim = 1)
        x = self.up3(x, xcat2)
        xcat1 = torch.cat((xa1,xb1),dim = 1)
        x = self.up4(x, xcat1)

        logits = self.outc(x)
        
        return logits
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

class SiamUnet_diff_Full(nn.Module):
    """SiamUnet_diff segmentation network."""

    def __init__(self, input_nbr, label_nbr):
        super(SiamUnet_diff_Full, self).__init__()

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
    
class SiamUnet_diff(nn.Module):
    """SiamUnet_diff segmentation network."""

    def __init__(self, input_nbr, label_nbr):
        super(SiamUnet_diff, self).__init__()

        n1 = 16
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