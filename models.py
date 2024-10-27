import torch
import torch.nn as nn
from torch.nn import init
from typing import Type
from torch import Tensor


#from __future__ import absolute_import, print_function

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_5(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_9(nn.Module):
    def __init_(self, ch_in, ch_out):
        super(conv_block_9, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=9, stride=1, padding=4, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=9, stride=1, padding=4, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_3_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_3_1, self).__init__()

        #self.conv_1 = conv_block_1(ch_in, ch_out)
        #self.conv_2 = conv_block_2(ch_in, ch_out)
        self.conv_3 = conv_block_3(ch_in, ch_out)
        #self.conv_5 = conv_block_5(ch_in, ch_out)
        self.conv_7 = conv_block_7(ch_in, ch_out)
        #self.conv_9 = conv_block_9(ch_in, ch_out)

        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        #x1 = self.conv_1(x)
        #x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        #x5 = self.conv_5(x)
        x7 = self.conv_7(x)
        #x9 = self.conv_9(x)

        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)

        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear = False):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_in),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x):

        x = self.up(x)

        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net, self).__init__()

        filters_number = [32, 64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv2 = conv_block(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv3 = conv_block(ch_in=filters_number[1], ch_out=filters_number[2])
        self.Conv4 = conv_block(ch_in=filters_number[2], ch_out=filters_number[3])
        self.Conv5 = conv_block(ch_in=filters_number[3], ch_out=filters_number[4])

        self.Up5 = up_conv(ch_in=filters_number[4], ch_out=filters_number[3])
        self.Up_conv5 = conv_block(ch_in=filters_number[4], ch_out=filters_number[3])

        self.Up4 = up_conv(ch_in=filters_number[3], ch_out=filters_number[2])
        self.Up_conv4 = conv_block(ch_in=filters_number[3], ch_out=filters_number[2])

        self.Up3 = up_conv(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Up_conv3 = conv_block(ch_in=filters_number[2], ch_out=filters_number[1])

        self.Up2 = up_conv(ch_in=filters_number[1], ch_out=filters_number[0])
        self.Up_conv2 = conv_block(ch_in=filters_number[1], ch_out=filters_number[0])

        self.Conv_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)
        
        #self.bottleneck_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4= self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class MSU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(MSU_Net, self).__init__()

        filters_number = [32, 64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_3_1(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv2 = conv_3_1(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv3 = conv_3_1(ch_in=filters_number[1], ch_out=filters_number[2])
        self.Conv4 = conv_3_1(ch_in=filters_number[2], ch_out=filters_number[3])
        self.Conv5 = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[4])

        self.Up5 = up_conv(ch_in=filters_number[4], ch_out=filters_number[3])
        self.Up_conv5 = conv_3_1(ch_in=filters_number[4], ch_out=filters_number[3])

        self.Up4 = up_conv(ch_in=filters_number[3], ch_out=filters_number[2])
        self.Up_conv4 = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[2])

        self.Up3 = up_conv(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Up_conv3 = conv_3_1(ch_in=filters_number[2], ch_out=filters_number[1])

        self.Up2 = up_conv(ch_in=filters_number[1], ch_out=filters_number[0])
        self.Up_conv2 = conv_3_1(ch_in=filters_number[1], ch_out=filters_number[0])

        self.Conv_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)
        
        #self.dropout = nn.Dropout2d(p=0.5)
        
        #self.fc1 = nn.Linear(1600*1600*1, 1)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 35)
        
        
    
    def forward(self, x):

        x1 = self.Conv1(x)
        #print(1)
        x2 = self.Maxpool(x1)
        #print(2)
        x2 = self.Conv2(x2)
        #print(3)
        x3 = self.Maxpool(x2)
        #print(4)
        x3 = self.Conv3(x3)
        #print(5)

        x4 = self.Maxpool(x3)
        #print(6)

        x4= self.Conv4(x4)
        #print(7)

        x5 = self.Maxpool(x4)
        #print(1)

        x5 = self.Conv5(x5)
        #print(2)

        d5 = self.Up5(x5)
        #print(3)

        d5 = torch.cat((x4, d5), dim=1)
        #d5 = self.dropout(d5)
        d5 = self.Up_conv5(d5)
        #print(4)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        #print(5)
        d4 = self.Up_conv4(d4)
        #print(6)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        #d3 = self.dropout(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        #d2 = self.dropout(d2)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        #print("final")
        #d1 = self.dropout(d1)
        
        #d1 = torch.flatten(d1, 1) # flatten all dimensions except batch
        #d1 = nn.functional.relu(self.fc1(d1))
        #d1 = nn.functional.relu(self.fc2(d1))
        #d1 = self.fc3(d1)
        
        return d1
    
    
    
    
    
    
class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out
    
    
    
    
    
class ResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 2
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc = nn.Linear(512*self.expansion, num_classes)
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        #x = self.dropout(x)
        x = self.layer2(x)
        #x = self.dropout(x)
        x = self.layer3(x)
        #x = self.dropout(x)
        x = self.layer4(x)
        #x = self.dropout(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    
    
    
    
    
    
    
    
    



class ConvReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, depth=2, kernel_size=3, padding=1) -> None:
        super(EncoderBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(ConvReLU(in_c if i == 0 else out_c, out_c, kernel_size, padding))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #self.pool_pad = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True, padding=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x, ind = self.pool(x)
        
        return x, ind
    

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, depth=2, kernel_size=3, padding=1, classification=False) -> None:
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == depth - 1 and classification:
                self.layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding))
            elif i == depth - 1:
                self.layers.append(ConvReLU(in_c, out_c, kernel_size=kernel_size, padding=padding))
            else:
                self.layers.append(ConvReLU(in_c, in_c, kernel_size=kernel_size, padding=padding))

    def forward(self, x, ind):
        x = self.unpool(x, ind)
        for layer in self.layers:
            x = layer(x)
        return x

class SegNet(nn.Module): 
    def __init__(self, in_channels=3, out_channels=1, features=64) -> None:
        super(SegNet, self).__init__()

        # Encoder
        self.enc0 = EncoderBlock(in_channels, features)
        self.enc1 = EncoderBlock(features, features * 2)
        self.enc2 = EncoderBlock(features * 2, features * 4, depth=3)
        self.enc3 = EncoderBlock(features * 4, features * 8, depth=3)

        # Bottleneck
        self.bottleneck_enc = EncoderBlock(features * 8, features * 8, depth=3) 
        self.bottleneck_dec = DecoderBlock(features * 8, features * 8, depth=3) 

        # Decoder
        self.dec0 = DecoderBlock(features * 8, features * 4, depth=3)
        self.dec1 = DecoderBlock(features * 4, features * 2, depth=3)
        self.dec2 = DecoderBlock(features * 2, features)
        self.dec3 = DecoderBlock(features, out_channels, classification=True) # No activation

    def forward(self, x):
        # encoder
        e0, ind0 = self.enc0(x) 
        e1, ind1 = self.enc1(e0) 
        e2, ind2 = self.enc2(e1) 
        e3, ind3 = self.enc3(e2)

        # bottleneck
        b0, indb = self.bottleneck_enc(e3)      
        b1 = self.bottleneck_dec(b0, indb)

        
        if e3.shape[2] != b1.shape[2]:
            size = torch.Size([ b1[0,0,:,:].shape[0] +1, b1[0,0,:,:].shape[1] ])
            #print(size)
            b1 = transforms.Resize(size = size, antialias=True )( b1 )
            ind3 = transforms.Resize(size = size, antialias=True )( ind3 )

        if e3.shape[3] != b1.shape[3]:
            size = torch.Size([ b1[0,0,:,:].shape[0], b1[0,0,:,:].shape[1]+1 ])
           # print(size)
            b1 = transforms.Resize(size = size, antialias=True )( b1 )
            ind3 = transforms.Resize(size = size, antialias=True )( ind3 )

        # decoder
        d0 = self.dec0(b1, ind3)
        d1 = self.dec1(d0, ind2)
        d2 = self.dec2(d1, ind1)

        # classification layer
        output = self.dec3(d2, ind0)  
        return output


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    