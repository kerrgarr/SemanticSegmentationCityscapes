import torch
import torch.nn.functional as F
from .utils import N_CLASSES
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
from torchvision import models
import torch.utils.model_zoo as model_zoo
import math
import sys, time, os, warnings 


##############################################################################################
###### DeepLabv3_plus
###### Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ArXiv. https://arxiv.org/pdf/1802.02611.pdf
###### References: jfzhang95. (2018). PyTorch DeepLab-XCeption. GitHub. https://github.com/jfzhang95/pytorch-deeplab-xception
##############################################################################################

class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * 4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(torch.nn.Module):
    # model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained) #=pretrained)
    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 16
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = torch.nn.Conv2d(nInputChannels, 16, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 128, blocks=blocks, stride=strides[3], rate=rates[3])

        #self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return torch.nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet101(nInputChannels=3, os=16, pretrained=False):
    model_resnet = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained) #=pretrained)
    return model_resnet


class ASPP_module(torch.nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = torch.nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU()

       # self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(torch.nn.Module):
    def __init__(self, nInputChannels=3, n_classes=N_CLASSES, os=16, pretrained=False, _print=False):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet101(nInputChannels, os, pretrained) #=pretrained)

        # ASPP
        if os == 16: ## output_stride
            rates = [1, 6, 12, 18]
        elif os == 8: ## output_stride
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(512, 64, rate=rates[0])
        self.aspp2 = ASPP_module(512, 64, rate=rates[1])
        self.aspp3 = ASPP_module(512, 64, rate=rates[2])
        self.aspp4 = ASPP_module(512, 64, rate=rates[3])

        self.relu = torch.nn.ReLU()

        self.global_avg_pool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)),
                                             torch.nn.Conv2d(512, 64, 1, stride=1, bias=False),
                                             torch.nn.BatchNorm2d(64),
                                             torch.nn.ReLU())

        self.conv1 = torch.nn.Conv2d(320, 64, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = torch.nn.Conv2d(64, 48, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(48)

        self.last_conv = torch.nn.Sequential(torch.nn.Conv2d(112, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       torch.nn.BatchNorm2d(64),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       torch.nn.BatchNorm2d(64),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv2d(64, n_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        #print("last conv", x.shape)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


####################################################################################################
###                                     U-Net
### Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. 
### International Conference on Medical Image Computing and Computer-Assisted Intervention, 234–241.
### References: https://github.com/dvssajay/The-Ikshana-Hypothesis-of-Human-Scene Understanding/blob/main/Cityscapes%20Baseline%20Experiments/U-Net/UNet-1/U-Net-1.py
### milesial. (2021). Pytorch-Unet. Github. https://github.com/milesial/Pytorch-UNet
##############################################################################################

class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(torch.nn.Module):
    def __init__(self, n_channels=3, n_classes=N_CLASSES, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        #print("0",x.shape)
        x1 = self.inc(x)
        #print("1",x1.shape)
        x2 = self.down1(x1)
        #print("2",x2.shape)
        x3 = self.down2(x2)
        #print("3",x3.shape)
        x4 = self.down3(x3)
        #print("4",x4.shape)
        x5 = self.down4(x4)
        #print("5",x5.shape)
        x = self.up1(x5, x4)
        #print("up1",x.shape)
        x = self.up2(x, x3)
        #print("up2",x.shape)
        x = self.up3(x, x2)
        #print("up3",x.shape)
        x = self.up4(x, x1)
        #print("up4",x.shape)
        logits = self.outc(x)
        #print("out",logits.shape)
        return logits
    

##############################################################################################################################
##### My FCN #################################################################################################################

def down_conv(small_channels, big_channels, pad):   ### contracting block
    return torch.nn.Sequential(
        torch.nn.Conv2d(small_channels, big_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(big_channels),
        torch.nn.Conv2d(big_channels, big_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(big_channels)
    )   ## consider stride = 2

def up_conv(big_channels, small_channels, pad):
    return torch.nn.Sequential(
        torch.nn.Conv2d(big_channels, small_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(small_channels),
        torch.nn.Conv2d(small_channels, small_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(small_channels)
    )


class my_FCN(torch.nn.Module):
    def crop(self, a, b):
        ## a, b tensor shape = [batch, channel, H, W]
        Ha = a.size()[2]
        Wa = a.size()[3]
        Hb = b.size()[2]
        Wb = b.size()[3]

        adapt = torch.nn.AdaptiveMaxPool2d((Ha,Wa))
        crop_b = adapt(b) 
            
        return crop_b    
   
    
    def __init__(self):
        super().__init__()

        self.relu    = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, ceil_mode=True)         
        self.mean = torch.Tensor([0.5, 0.5, 0.5])
        self.std = torch.Tensor([0.25, 0.25, 0.25])
        
        a = 32
        b = a*2 #64
        c = b*2 #128
        d = c*2 #256
        
        n_class = N_CLASSES
        
        self.conv_down1 = down_conv(3, a, 1) # 3 --> 32
        self.conv_down2 = down_conv(a, b, 1)  # 32 --> 64
        self.conv_down3 = down_conv(b, c, 1)  # 64 --> 128
        self.conv_down4 = down_conv(c, d, 1)  # 128 --> 256
        
        self.bottleneck = torch.nn.ConvTranspose2d(d, c, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.conv_up3 = up_conv(c, b, 1)  # 128 --> 64
        self.upsample3 = torch.nn.ConvTranspose2d(b, a, kernel_size=3, stride=2, padding=1, output_padding=1)   
                 
        self.classifier = torch.nn.Conv2d(a, n_class, kernel_size=1) 
        
    
    def forward(self, x):
        H = x.shape[2]
        W = x.shape[3]
        z = (x - self.mean[None, :, None, None].to(x.device)) / self.std[None, :, None, None].to(x.device)
        #################### DOWN / ENCODER #############################
        conv1 =  self.conv_down1(z)   # 3 --> 32
        mx1 = self.maxpool(conv1)
        conv2 =  self.conv_down2(mx1)  # 32 --> 64
        mx2 = self.maxpool(conv2) 
        conv3 =  self.conv_down3(mx2) # 64 --> 128  
        mx3 = self.maxpool(conv3) 
        conv4 =  self.conv_down4(conv3) # 128 --> 256  ################### CHANGED THIS

        ########################### BOTTLENECK #############################
        score = self.bottleneck(conv4)  # 256 --> 128
       
        ######################### UP/DECODER #######################
        crop_conv3 = self.crop(score, conv3)    
        score = score + crop_conv3   ### add 128 
        
        ##########################
        score = self.conv_up3(score)  # 128 --> 64
        score = self.upsample3(score)  # 64 --> 32     
        crop_conv1 = self.crop(score, conv1)   
        score = score + crop_conv1   ### add 32           
        
        ############################
        score = self.classifier(score) 
        out = torch.nn.functional.interpolate(score, size=(H,W))
        out = out[:, :, :H, :W]
        return out  




###############################################################################################################################

model_factory = {
    'my_fcn': my_FCN, 
    'unet': UNet,
    'deeplab': DeepLabv3_plus
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
