import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from  models.MPNCOV import MPNCOV
import torch.nn.functional as F
import torchvision
from .position_encoding import build_position_encoding
import copy

import re
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['pro_net']
       

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    
class PrtAttLayer(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def prt_interact(self,sem_prt):
        tgt2 = self.self_attn(sem_prt, sem_prt, value=sem_prt)[0]
        sem_prt = sem_prt + self.dropout1(tgt2)
        return sem_prt
        
    def prt_assign(self, sem_prt, vis_query):
        sem_prt = self.norm1(sem_prt)
        sem_prt2 = self.multihead_attn(query=sem_prt,
                                   key=vis_query,
                                   value=vis_query)[0]
        sem_prt = sem_prt + self.dropout2(sem_prt2)
        return sem_prt
    def prt_refine(self, sem_prt):
        sem_prt = self.norm2(sem_prt)
        sem_prt3 = self.linear2(self.dropout(self.activation(self.linear1(sem_prt))))
        sem_prt = sem_prt + self.dropout3(sem_prt3)
        return sem_prt

    def forward(self, in_prt, query):
        # sem_prt: 312*bs*c
        # vis_query: wh*bs*c
        in_prt = self.prt_interact(in_prt)
        in_prt = self.prt_assign(in_prt,query)
        in_prt = self.prt_refine(in_prt)
        return in_prt

class PrtCateLayer(nn.Module):
    def __init__(self, dim, n=1):
        super().__init__()
        
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        
        self.fc1 = nn.Linear(dim, dim//n)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Linear(dim//n, dim)
        
        self.activation = nn.ReLU()

    def prt_refine(self, prt):
        prt1 = self.linear2(self.activation(self.linear1(prt)))
        
        w = self.activation(self.fc1(prt1))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        prt = prt1# + prt1 * w 
        return prt

    def forward(self, in_prt, query):
        # in_prt: 200*64*c
        # query: 64*c
        out_prt = self.prt_refine(in_prt)
        return out_prt
        
class Model(nn.Module):
    def __init__(self, pretrained=True, args=None):
        super(Model, self).__init__()
        ''' default '''
        num_classes = args.num_classes
        is_fix = args.is_fix
        sf_size = args.sf_size
        self.sf =  torch.from_numpy(args.sf)
        self.args  = args
        feat_dim = 2048
        # position embedding
        self.args.position_embedding='sine'
        self.args.hidden_dim=2048
        self.pos_emb_func = build_position_encoding(self.args)
        
        ''' backbone net'''
        if args.backbone=='resnet101':
            self.backbone = torchvision.models.resnet101()
        elif args.backbone=='resnet50':
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

        ''' ZSR '''
        if args.n_enc==0:                                            
            self.zsr_vis_proj = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0,bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True),
            )
        else:
            nhead=8
            num_encoder_layers=3
            encoder_layer = TransformerEncoderLayer(d_model=feat_dim, nhead=nhead, dim_feedforward=feat_dim)
            self.zsr_vis_proj = TransformerEncoder(encoder_layer, num_encoder_layers)

        if self.args.n_dec==0:
            self.zsr_sem_proj = nn.Sequential(
                nn.Linear(sf_size,1024),
                nn.LeakyReLU(),
                nn.Linear(1024,feat_dim),
                nn.LeakyReLU(),
            )
            self.zsr_aux_cls = nn.Linear(feat_dim, num_classes)
        else:
            n_per_att = int(feat_dim/sf_size)
            n_att_emb = n_per_att*sf_size
            # att prt
            self.zsr_prt_emb_att = nn.Embedding(sf_size, feat_dim)
            self.zsr_prt_ref_att = _get_clones(PrtAttLayer(dim=feat_dim, nhead=8), args.n_dec)
            self.zsr_prt_proj_att = _get_clones(nn.Sequential(nn.Linear(feat_dim,n_per_att),nn.LeakyReLU()), args.n_dec)
            # cate prt
            self.zsr_prt_emb_cate = nn.Embedding(num_classes, n_att_emb)
            self.zsr_prt_ref_cate = _get_clones(PrtCateLayer(dim=n_att_emb, n=n_per_att), args.n_dec)
            
            # sem proj
            self.zsr_sem_proj = nn.Sequential(
                nn.Linear(sf_size,1024),
                nn.LeakyReLU(),
                nn.Linear(1024,n_att_emb),
                nn.LeakyReLU(),
            )
        
        ''' params ini '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if pretrained:
            if args.backbone=='resnet101':
                self.backbone.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))
            elif args.backbone=='resnet50':
                self.backbone.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
            
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

    def forward(self, x):
        # backbone
        last_conv = self.backbone(x)
        bs, c, h, w = last_conv.shape
        
        if self.args.n_enc==0:
            # wh*bs*c
            vis_query = self.zsr_vis_proj(last_conv).flatten(2).permute(2, 0, 1)
        else:
            # position embedding
            pos_emb = self.pos_emb_func(last_conv).flatten(2).permute(2, 0, 1)
            src = last_conv.flatten(2).permute(2, 0, 1)
            # encoder
            vis_query = self.zsr_vis_proj(src, pos=pos_emb)
        
        ''' Dual Prototype Refine Network'''
        if self.args.n_dec==0:
            x = vis_query.permute(1, 2, 0).view(bs, c, h, w)
            zsr_x = self.avgpool(x).view(bs,-1)
            #zsr_x = self.zsr_pred(x).sigmoid().squeeze()    
            zsr_w = self.zsr_sem_proj(self.sf.cuda())
            w_norm = F.normalize(zsr_w, p=2, dim=1)
            x_norm = F.normalize(zsr_x, p=2, dim=1)
            zsr_logit_att = [x_norm.mm(w_norm.permute(1,0))]
            zsr_logit_cate = [self.zsr_aux_cls(zsr_x)]
        else:
            # prt init
            prt_att_init = self.zsr_prt_emb_att.weight.unsqueeze(1).repeat(1, bs, 1).cuda()
            prt_cate_init = self.zsr_prt_emb_cate.weight.unsqueeze(1).repeat(1, bs, 1).cuda()
            # semantic projection
            zsr_w = self.zsr_sem_proj(self.sf.cuda())
            w_norm = F.normalize(zsr_w, p=2, dim=1)
            # attriute prototype refine
            prt = prt_att_init
            vis_att_query = []
            zsr_logit_att = []
            for prt_ref,prt_proj in zip(self.zsr_prt_ref_att,self.zsr_prt_proj_att):
                prt = prt_ref(prt, vis_query)
                zsr_x = prt_proj(prt).permute(1,0,2).flatten(1)
                vis_att_query.append(zsr_x)
                x_norm = F.normalize(zsr_x, p=2, dim=1)
                zsr_logit_att.append(x_norm.mm(w_norm.permute(1,0)))
            # category prototype refine
            prt = prt_cate_init
            zsr_logit_cate = []
            for prt_ref,query in zip(self.zsr_prt_ref_cate,vis_att_query):
                prt = prt_ref(prt,query)
                classifier = prt.permute(1,0,2)
                query = query.unsqueeze(2)
                logit = classifier.bmm(query).squeeze()
                zsr_logit_cate.append(logit)
        
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

        return (ood_logit,zsr_logit_att[-1],zsr_logit_cate[-1],zsr_logit_att,zsr_logit_cate),(ood_x,zsr_x)


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.sf =  torch.from_numpy(args.sf).cuda()
        self.cls_loss = nn.CrossEntropyLoss()#reduce=False)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, label, logits, feats):
        ood_logit = logits[0]
        zsr_logits_att = logits[3]
        zsr_logits_cate = logits[4]
        
        ''' ZSL Loss '''
        L_att = 0
        for logit in zsr_logits_att:
            idx = torch.arange(logit.size(0)).long()
            L_att += (1-logit[idx,label]).mean()/len(zsr_logits_att)
        L_cate = 0
        for logit in zsr_logits_cate:
            L_cate += self.cls_loss(logit,label)/len(zsr_logits_cate)
        
        L_zsr = L_att + self.args.L_cate*L_cate
        
        ''' OOD Loss '''
        L_ood = self.cls_loss(ood_logit,label)
        
        total_loss = L_zsr + self.args.L_ood*L_ood
		
        return total_loss,L_ood,L_zsr,L_att,L_cate
		
def pro_net(pretrained=True, loss_params=None, args=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Model(pretrained,args)
    loss_model = Loss(args)
    return model,loss_model
	
	
