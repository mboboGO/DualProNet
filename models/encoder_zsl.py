import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from  models.MPNCOV import MPNCOV
import torch.nn.functional as F
import torchvision
from .position_encoding import build_position_encoding
from .transformer import *


import re
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['encoder']
       
       
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
        
        
class Model(nn.Module):
    def __init__(self, pretrained=True, args=None):
        super(Model, self).__init__()
        ''' default '''
        num_classes = args.num_classes
        is_fix = args.is_fix
        sf_size = args.sf_size
        self.sf =  torch.from_numpy(args.sf).cuda()
        self.args  = args
        feat_dim = 2048
        # position embedding
        self.args.position_embedding='sine'
        self.args.hidden_dim=2048
        self.pos_emb_func = build_position_encoding(self.args)
        # att query
        self.query_embed = nn.Embedding(sf_size, feat_dim)
        
        ''' backbone net'''
        self.backbone = torchvision.models.resnet50()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if(is_fix):
            for p in self.parameters():
                p.requires_grad=False

        ''' Domain Detector '''
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ood_proj =  nn.Sequential(
                nn.Conv2d(feat_dim, 256, kernel_size=1, stride=1, padding=0,bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
        )
        self.ood_spatial =  nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
                nn.ReLU(inplace=True),   
                nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0,bias=False),
                nn.Sigmoid(),        
        )
        self.ood_channel =  nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(256, int(256/16), kernel_size=1, stride=1, padding=0,bias=False),
                nn.ReLU(inplace=True),   
                nn.Conv2d(int(256/16), 256, kernel_size=1, stride=1, padding=0,bias=False),
                nn.Sigmoid(),        
        )
        self.ood_classifier = nn.Linear(int(256*(256+1)/2), num_classes)

        ''' Transformer '''
        d_model=2048
        nhead=8
        dim_feedforward=2048
        num_encoder_layers=3
        num_decoder_layers=3
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.zsr_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        decoder_norm = nn.LayerNorm(d_model)
        self.zsr_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=True)
                                                                                    
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
            self.backbone.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

    def forward(self, x):
        # backbone
        last_conv = self.backbone(x)
        bs, c, h, w = last_conv.shape
        
        ''' ZSR encoder '''
        # position embedding
        pos_emb = self.pos_emb_func(last_conv).flatten(2).permute(2, 0, 1)
        src = last_conv.flatten(2).permute(2, 0, 1)
        # encoder
        memory = self.zsr_encoder(src, pos=pos_emb)
        
        ''' ZSR decoder 1 '''
        #query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        #tgt = torch.zeros_like(query_embed)
        #hs = self.zsr_decoder(tgt, memory, pos=pos_emb, query_pos=query_embed)
        
        memory = memory.permute(1, 2, 0).view(bs, c, h, w)
        #hs = hs.transpose(1, 2)
        
        # out
        x = self.zsr_pro(last_conv)
        zsr_x = self.avgpool(memory).view(bs,-1)
        #zsr_x = self.zsr_pred(x).sigmoid().squeeze()    
        zsr_w = self.zsr_sem(self.sf)
        w_norm = F.normalize(zsr_w, p=2, dim=1)
        x_norm = F.normalize(zsr_x, p=2, dim=1)
        zsr_logit = x_norm.mm(w_norm.permute(1,0))
        zsr_logit_aux = self.zsr_aux(zsr_x)

        
        ''' OOD Module '''
        x = self.ood_proj(last_conv)
        # att gen
        att1 = self.ood_spatial(x)
        att2 = self.ood_channel(x)
        # att1
        x1 = att2*x+x
        x1 = x1.view(x.size(0),x.size(1),-1)
        # att2
        x2 = att1*x+x
        x2 = x2.view(x.size(0),x.size(1),-1)
        # covariance pooling
        x1 = x1 - torch.mean(x1,dim=2,keepdim=True)
        x2 = x2 - torch.mean(x2,dim=2,keepdim=True)
        A = 1./x1.size(2)*x1.bmm(x2.transpose(1,2))
        # norm
        x = MPNCOV.SqrtmLayer(A, 5)
        x = MPNCOV.TriuvecLayer(x)
        ood_x = x.view(x.size(0), -1)
        # cls
        ood_logit = self.ood_classifier(ood_x)

        return (ood_logit,zsr_logit,zsr_logit_aux),(ood_x,zsr_x,memory)
  
		
class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.sf =  torch.from_numpy(args.sf).cuda()
        self.cls_loss = nn.CrossEntropyLoss()#reduce=False)
        self.mse_loss = nn.MSELoss()
        self.sigma = 0.5
        
    def forward(self, label, logits, feats):
        ood_logit = logits[0]
        zsr_logit = logits[1]
        zsr_logit_aux = logits[2]
        
        zsr_x = feats[1]
        
        ''' ZSL Loss '''
        idx = torch.arange(zsr_logit.size(0)).long()
        L_zsr = (1-zsr_logit[idx,label]).mean()
        L_aux = self.cls_loss(zsr_logit_aux,label)
        
        #L_mse = self.mse_loss(zsr_x,self.sf[label,:])
        
        L_zsr = L_zsr + L_aux
        
        
        ''' OOD Loss '''
        L_ood = self.cls_loss(ood_logit,label)
        
        total_loss = L_zsr + self.args.L_ood*L_ood
		
        return total_loss,L_zsr,L_ood,L_aux
		
def encoder(pretrained=True, loss_params=None, args=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Model(pretrained,args)
    loss_model = Loss(args)
    return model,loss_model
	
	
