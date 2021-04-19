import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import torchvision


import re
from torch.utils.model_zoo import load_url as load_state_dict_from_url

class BaseS2V(nn.Module):
    def __init__(self, pretrained=True, args=None):
        super(BaseS2V, self).__init__()
        ''' default '''
        num_classes = args.num_classes
        is_fix = args.is_fix
        sf_size = args.sf_size
        self.sf =  torch.from_numpy(args.sf)
        self.args  = args
        feat_dim = 2048
        
        ''' backbone net'''
        if args.backbone=='r50':
            self.backbone = torchvision.models.resnet50()
        elif args.backbone=='r101':
            self.backbone = torchvision.models.resnet101()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        if(is_fix):
            for p in self.parameters():
                p.requires_grad=False

        ''' semantic-visual alignment '''                                                                   
        self.zsr_pro = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.zsr_sem = nn.Sequential(
            nn.Linear(sf_size,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,feat_dim),
            nn.LeakyReLU(),
        )
        self.zsr_aux = nn.Linear(feat_dim, num_classes)
        
        ''' params ini '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if pretrained:
            print('==> Pretrained!')
            if args.backbone=='r50':
                self.backbone.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
            elif args.backbone=='r101':
                self.backbone.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

    def forward(self, x):
        # backbone
        last_conv = self.backbone(x)
        bs, c, h, w = last_conv.shape
        
        ''' ZSR '''
        x = self.zsr_pro(last_conv)
        zsr_x = self.avgpool(x).view(bs,-1) 
        zsr_w = self.zsr_sem(self.sf.cuda())
        w_norm = F.normalize(zsr_w, p=2, dim=1)
        x_norm = F.normalize(zsr_x, p=2, dim=1)
        logit = x_norm.mm(w_norm.permute(1,0))
        logit_aux = self.zsr_aux(zsr_x)

        return [logit],[logit_aux]

