from __future__ import absolute_import, division, print_function

import math
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# model_urls = {
#     'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed3d.pth',
# }

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # change
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

#-----------------------------------------------------------------#
#   使用Renset50作为主干特征提取网络，最终会获得一个
#   8x8x8x2048的有效特征层
#-----------------------------------------------------------------#
class ResNet_50(nn.Module):
    def __init__(self, block, layers, num_classes=1):  # ResNet_50(Bottleneck, [3, 4, 6, 3])
        super(ResNet_50, self).__init__()

        self.inplanes = 64  # conv1 的输出维度
        # 128,128,128,1 -> 256,256,64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # 256x256x64 -> 128x128x64  H/2, D/2, C stay
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1, ceil_mode=True) # change

        # 128x128x64 -> 128x128x256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 128x128x256 -> 64x64x512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # 64x64x512 -> 32x32x1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  

        # 32x32x1024 -> 16x16x2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool3d(7)   # * 将每张特征图大小->(1,1), 则经过池化层后的输出维度维=通道数
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks_num, stride=1):
        '''
            block:堆叠的基本块
            channel:每个stage中堆叠模块的第一个卷积的卷积核个数,对resnet50分别是:64, 128, 256, 512
            block_num:当期stage堆叠block的个数
            stride:默认的卷积步长
        '''
        downsample = None # * 控制shortcut路径的
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, 
                          planes * block.expansion,
                          kernel_size=1, 
                          stride=stride, 
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks_num):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)  # "*" 的作用是为了将list转换为非关键字参数传入

    def forward(self, x):
        # origin x : torch.Size([1, 1, 128, 128, 128])
        x = self.conv1(x)
        # embed() # torch.Size([1, 64, 64, 64, 64])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # embed() # torch.Size([1, 64, 64, 64, 64])


        x = self.layer1(x)
        # embed() # torch.Size([1, 256, 64, 64, 64])
        x = self.layer2(x)
        # embed() #  torch.Size([1, 512, 32, 32, 32])
        x = self.layer3(x)
        # embed() # torch.Size([1, 1024, 16, 16, 16])
        x = self.layer4(x)
        # embed() # torch.Size([1, 2048, 8, 8, 8])
        
        # x = self.avgpool(x)
        # # torch.Size([1, 2048, 1, 1, 1])
        # x = x.view(x.size(0), -1)
        # # torch.Size([1, 2048])
        # x = self.fc(x)
        # # torch.Size([1, 1])

        return x



class resnet50_Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(resnet50_Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False
        
        #----------------------------------------------------------#
        #   2048, 8, 8, 8 -> 512, 16, 16, 16 -> 256, 32, 32, 32 -> 128, 64, 64, 64 -> 64, 128, 128, 128
        #   利用ConvTranspose3d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        #----------------------------------------------------------#
        self.deconv_layers = self._make_deconv_layer(
            num_layers=4,
            num_filters=[512, 256, 128, 64],
            num_kernels=[4, 4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i] # channel number

            layers.append(
                nn.ConvTranspose3d(  
                    in_channels=self.inplanes,
                    out_channels=planes, 
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm3d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

            # 这里建造了four层反卷积layers, 每一个layer都包含一个卷积，一个bn，和一个relu
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)
    





class resnet50_Head(nn.Module):
    def __init__(self, num_classes=1, channel=64, bn_momentum=0.1):
        super(resnet50_Head, self).__init__()
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        # 热力图预测部分
        self.cls_head = nn.Sequential(
            nn.Conv3d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv3d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, 3,
                      kernel_size=1, stride=1, padding=0))

        # 中心点预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv3d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, 3,
                      kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        return hm, wh, offset



def resnet50():

    return ResNet_50(Bottleneck, [3, 4, 6, 3])
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir = 'model_data/')
    #     model.load_state_dict(state_dict)
    #----------------------------------------------------------#
    #   获取特征提取部分
    #----------------------------------------------------------#
    # features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4])
    # features = nn.Sequential(*features)
    # return features

if __name__ == '__main__':
    import torch
    from IPython import embed

    input = torch.randn(1, 1, 128, 128, 128)
    result = resnet50()
    embed()
