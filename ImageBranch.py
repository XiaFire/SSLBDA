#
# 图像复原任务的网络结构定义与损失函数
#
import torch
from torch import nn
from torch.nn import functional as F

class Fusion(nn.Module):
    '''
    Fusion层：进行特征融合，包括一个 3x3 卷积层跟一个 ReLU 激活层
    '''
    def __init__(self, in_channels, out_channels):
        super(Fusion, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
            )
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_relu(x)
        return x

class ImageDecoderCNN(nn.Module):
    '''
    图像复原任务的解码器, CNN
    '''
    def __init__(self, channels=3, hidden_size = [512, 256, 128, 64], sample_ind=[4,3,2,1,0]):
        super(ImageDecoderCNN, self).__init__()
        assert len(hidden_size) == len(sample_ind)-1, "The length of hidden_size should be smaller than the length of sample_ind by 1"
        self.sample_ind = sample_ind
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        self.upx4 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=False)
        self.fusion = []
        for in_channel, out_channel in zip(hidden_size[:-1], hidden_size[1:]):
            self.fusion.append(Fusion(in_channel+out_channel, out_channel).cuda())
        self.conv1 = nn.Conv2d(hidden_size[3], channels, kernel_size=1)
    
    def forward(self, out):
        x = out[self.sample_ind[0]]
        for ind, fusion in zip(self.sample_ind[1:], self.fusion):
            x = self.up(x)
            x = fusion(x, out[ind])
        x = self.conv1(x) 
        x = self.upx4(x)
        return x

class ImageDecoderVit(nn.Module):
    '''
    图像复原任务的解码器
    '''
    def __init__(self, channels=3, hidden_size = [384, 384, 384], sample_ind = [11, 8, 5, 2]):
        super(ImageDecoderVit, self).__init__()
        assert len(hidden_size) == len(sample_ind)-1, "The length of hidden_size should be smaller than the length of sample_ind by 1"
        self.hidden_size = hidden_size
        self.sample_ind = sample_ind
        self.fusion = []
        for in_channel, out_channel in zip(hidden_size[:-1], hidden_size[1:]):
            self.fusion.append(Fusion(in_channel+out_channel, out_channel).cuda())
        self.conv1 = nn.Conv2d(hidden_size[-1], channels, kernel_size=1)
    
    def forward(self, all_out):
        outs = []
        for ind in self.sample_ind:
            o = all_out[ind]
            o = o[:,1:,:]
            o = o.transpose(1,2)
            patch_size = int((o.size(2))**0.5)
            o = o.reshape(o.size(0), o.size(1), patch_size, patch_size)
            outs.append(o)
        x = outs[0]
        for out, fusion in zip(outs[1:], self.fusion):
            x = fusion(x, out)
        x = self.conv1(x)
        return x

class ImageLoss(nn.Module):
    '''
    图像复原任务的损失
    '''
    def __init__(self, ncrops):
        super().__init__()
        self.ncrops = ncrops
    def forward(self, img_in, img_out):
        """
        L1-Loss
        n_crops = (B1 + B2) / batch_size 一组的图片数 10
        """
        total_loss = 0
        for img, out in zip(img_in, img_out):
            total_loss += F.l1_loss(img, out) 
        total_loss /= self.ncrops
        return total_loss

