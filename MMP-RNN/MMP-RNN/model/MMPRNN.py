# +
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .UNet import UNet
# -

from .arches import conv1x1, conv3x3, conv5x5, actFunc, make_blocks
from data.utils import normalize_reverse


# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='gelu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Channel attention layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x*y, y

class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False) #False
        self.sigmoid = nn.Sigmoid()
        self.conv2 = conv1x1(1,1)
    def forward(self, x, bm):
        #print(x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #x = torch.cat([avg_out, max_out], dim=1)
        y = avg_out
        #bm = self.conv2(bm)
        y = torch.cat([y, bm], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        #print(y.shape)
        return x*y


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='gelu',do_attention_ca=False,do_attention_sa=False):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        self.do_attention_ca = do_attention_ca
        self.do_attention_sa=do_attention_sa
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)
        if do_attention_ca:
            self.ca = CALayer(in_channels_)
        if do_attention_sa:
            self.sa = SALayer()

    def forward(self, x):
        out = self.dense_layers(x)
        if self.do_attention_ca:
            out,_ = self.ca(out)
        if self.do_attention_sa:
            out_ = out
            out = self.sa(out_)
            out+=out_
        
        out = self.conv1x1(out)
        out += x
        return out


# Middle network of residual dense blocks
class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks, activation='gelu', do_attention_ca=False,do_attention_sa=False):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer, activation,do_attention_ca=do_attention_ca,do_attention_sa=do_attention_sa))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out


# DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, out_channels, num_layer, activation='gelu'):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, out_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out


# DownSampling module
class RB_DS(nn.Module):
    def __init__(self, in_channels, out_channels, activation='gelu'):
        super(RB_DS, self).__init__()
        self.rb1 = ResBlock(in_channels, activation)
        self.rb2 = ResBlock(in_channels, activation)
        self.down_sampling = conv5x5(in_channels, out_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rb1(x)
        x = self.rb2(x)
        out = self.down_sampling(x)

        return out


# Global spatio-temporal attention module
class GSA(nn.Module):
    def __init__(self, para):
        super(GSA, self).__init__()
        self.n_feats = para.n_features
        self.center = para.past_frames
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.F_f = nn.Sequential(
            nn.Linear(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            actFunc(para.activation),
            nn.Linear(4 * (5 * self.n_feats), 2 * (5 * self.n_feats)),
            nn.Sigmoid()
        )
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            conv1x1(4 * (5 * self.n_feats), 2 * (5 * self.n_feats))
        )
        # condense layer
        self.condense = conv1x1(2 * (5 * self.n_feats), 5 * self.n_feats)
        # fusion layer
        self.fusion = conv1x1(self.related_f * (5 * self.n_feats), self.related_f * (5 * self.n_feats))

    def forward(self, hs):
        # hs: [(n=4,c=80,h=64,w=64), ..., (n,c,h,w)]
        self.nframes = len(hs)
        f_ref = hs[self.center]
        cor_l = []
        for i in range(self.nframes):
            if i != self.center:
                cor = torch.cat([f_ref, hs[i]], dim=1)
                w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
                if len(w.shape) == 1:
                    w = w.unsqueeze(dim=0)
                w = self.F_f(w)
                w = w.reshape(*w.shape, 1, 1)
                cor = self.F_p(cor)
                cor = self.condense(w * cor)
                cor_l.append(cor)
        cor_l.append(f_ref)
        out = self.fusion(torch.cat(cor_l, dim=1))

        return out


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, in_chs, activation='gelu', batch_norm=False):
        super(ResBlock, self).__init__()
        op = []
        for i in range(2):
            op.append(conv3x3(in_chs, in_chs))
            if batch_norm:
                op.append(nn.BatchNorm2d(in_chs))
            if i == 0:
                op.append(actFunc(activation))
        self.main_branch = nn.Sequential(*op)

    def forward(self, x):
        out = self.main_branch(x)
        out += x
        return out


class EBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act='relu'):
        super(EBlock, self).__init__()
        self.conv = nn.Sequential(conv5x5(in_channels, out_channels, stride), actFunc(act))
        self.resblock_stack = make_blocks(ResBlock, num_basic_block=4, in_chs=out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.resblock_stack(out)
        return out


# MMAM
class MMAMLayer(nn.Module):
    def __init__(self, para):
        super(MMAMLayer, self).__init__()
        self.n_feats = para.n_features
        self.MMAM_conv0 = nn.Conv2d(1, self.n_feats // 2, 1)
        self.MMAM_conv1 = nn.Conv2d(self.n_feats //2, self.n_feats, 1)
       

    def forward(self, x, y):
        scale = self.MMAM_conv1(F.leaky_relu(self.MMAM_conv0(y), 0.1, inplace=True))
        return x * (scale+1)

class Skip(nn.Module):
    def __init__(self, para):
        super(Skip, self).__init__()
    def forward(self, x, y):
        return x

class CAT(nn.Module):
    def __init__(self, para):
        super(CAT, self).__init__()
        self.frames = para.future_frames+para.past_frames+1
        self.n_feats = para.n_features
        self.center = para.past_frames
        self.fusion = conv1x1(self.frames * (5 * self.n_feats), (5 * self.n_feats))
        self.ca = CALayer(self.frames * (5 * self.n_feats))
    def forward(self, hs):
        out = torch.cat(hs, dim=1)
        #out,_ = self.ca(out)
        out = self.fusion(out)
        return out


# RDB-based RNN cell
class RDBCell(nn.Module):
    def __init__(self, para):
        super(RDBCell, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks_a
        self.F_B0 = conv5x5(3, self.n_feats, stride=1)
        self.MMAM1 = MMAMLayer(para)
        #self.CT1 = ContrastLayer(para)
        self.F_B1 = RDB_DS(in_channels=1*self.n_feats, growthRate=int(self.n_feats * 2 / 2), out_channels = 2*self.n_feats, num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), out_channels = 2 * self.n_feats, num_layer=3,activation=self.activation)
        #RB_DS(in_channels=1*self.n_feats, out_channels = 2*self.n_feats, activation=self.activation) EBlock((1 + 2 +2) * self.n_feats, (1 + 2 +2) * self.n_feats, 1)
        self.F_R = RDNet(in_channels=(1 + 2 +2) * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks, activation=self.activation, do_attention_ca=False, do_attention_sa=False)  # in: 80
        # F_h: hidden state part
        self.F_h = nn.Sequential(
            conv3x3((1 + 2 +2) * self.n_feats, self.n_feats),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation),
            conv3x3(self.n_feats, self.n_feats)
        )
        
        self.prior =UNet()
        
        checkpoint = torch.load("./model_best.pth.tar") #, map_location=lambda storage, loc: storage.cuda()
        self.prior.load_state_dict(checkpoint['state_dict'])
        self.prior.eval()
        
        for param in self.prior.parameters():
            param.requires_grad = False

        

    def forward(self, x, s_last, mid_last):
        x0 = x
        out = self.F_B0(x0)
        mmp = self.prior(x0)
        
        out = self.MMAM1(out,mmp)
        
        out = self.F_B1(out)
        out = self.F_B2(out)
        
        mid = out
        
        out = torch.cat([out, mid_last, s_last], dim=1)
        
        out = self.F_R(out)
        s = self.F_h(out)
       
        return out, s, mid


# Reconstructor
class Reconstructor(nn.Module):
    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.para = para
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks_b
        self.D_Net = RDNet(in_channels=(1+2+2)*self.n_feats, growthRate=2*self.n_feats, num_layer=3, num_blocks=self.n_blocks, do_attention_ca=False, do_attention_sa=False)
        self.model = nn.Sequential(
            nn.ConvTranspose2d((5 * self.n_feats), 2 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv5x5(self.n_feats, 3, stride=1)
        )
        #self.sa = SALayer()

    def forward(self, x):
        x = self.D_Net(x)
        #x = self.sa(x,bm) 
        return self.model(x)


class Model(nn.Module):
    """
    new model
    """
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.ds_ratio = 4
        self.device = torch.device('cuda')
        self.cell = RDBCell(para)
        self.recons = Reconstructor(para)
        self.fusion = CAT(para)
        self.do_skip = para.do_skip
        self.centralize = para.centralize
        self.normalize = para.normalize
        if self.do_skip == True:
            self.skip = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=9, stride =1, padding=4, bias=True)
        
    def forward(self, x, profile_flag=False):
        if profile_flag:
            return self.profile_forward(x)
        outputs, hs = [], []
        batch_size, frames, channels, height, width = x.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        # forward h structure: (batch_size, channel, height, width)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        mid =  torch.zeros(batch_size, 2*self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            h, s, mid = self.cell(x[:, i, :, :, :], s, mid)
            hs.append(h)
        for i in range(self.num_fb, frames - self.num_ff):
           
            
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            
            out = self.recons(out)
            if self.do_skip == True:
                skip = self.skip(x[:, i, 0:3, :, :])
                out = out + skip
            outputs.append(out.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1) 

    # For calculating GMACs
    def profile_forward(self, x):
        outputs, hs = [], []
        batch_size, frames, channels, height, width = x.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        mid = torch.zeros(batch_size, 2*self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            h, s, mid = self.cell(x[:, i, :, :, :], s, mid)
            hs.append(h)
        for i in range(self.num_fb + self.num_ff):
            hs.append(torch.randn(*h.shape).to(self.device))
        x_skip = torch.zeros(batch_size, frames + self.num_fb + self.num_ff,channels, height, width).to(self.device)    
        for i in range(self.num_fb, frames + self.num_fb):
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            if self.do_skip == True:
                skip = self.skip((x_skip[:, i, 0:3, :, :]))
                out = out + skip

            outputs.append(out.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1)


def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True

    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)
   
    return flops / seq_length, params
