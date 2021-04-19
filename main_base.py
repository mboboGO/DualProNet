#from __future__ import print_function
import argparse
import os
import random
import shutil
import time
import warnings
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import sys
import torchvision.transforms as transforms
import datasets
import models
from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
print(model_names)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
''' opt for data '''
parser.add_argument('--data','-d', metavar='DATA', default='cub',
                    help='dataset')
parser.add_argument('--data-name', default='cub', type=str, help='dataset')     
parser.add_argument('--flipping-test', action='store_true',help='flipping test.')
# multi-gpu
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')  
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--is-syncbn', action='store_true', help='')
''' opt for optimizer'''
parser.add_argument('--epochs', default=180, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_decay', default=30, type=int,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--is_fix', dest='is_fix', action='store_true',
                    help='is_fix.')
''' opt for model '''
parser.add_argument('--backbone', default='resnet101', help='')
parser.add_argument('--model', default='dvbe', help='')
''' opt for others '''
parser.add_argument('--save-path', '-s', metavar='SAVE', default='',
                    help='saving path')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

             
best_prec1 = 0

def main():
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    ''' random seed '''
    if args.seed is not None:
        random.seed(args.seed)
    else:
        args.seed = random.randint(1, 10000)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    main_worker(ngpus_per_node, args)
    
def main_worker(ngpus_per_node, args):
    global best_prec1
    
    ''' multi-gpu '''
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        
    ''' logger '''
    if args.distributed:
        args.logger = setup_logger(output=args.save_path, distributed_rank=dist.get_rank(), name="model")
    else:
        args.logger = setup_logger(output=args.save_path, phase='ft')
    args.logger.info(args)

    ''' seed '''
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    args.logger.info('==> random seed: {}'.format(args.seed))

    ''' Data Load '''
    # data load info
    data_info = h5py.File(os.path.join('./data',args.data_name,'data_info.h5'), 'r')
    nc = data_info['all_att'][...].shape[0]
    sf_size = data_info['all_att'][...].shape[1]
    semantic_data = {'seen_class':data_info['seen_class'][...],
                     'unseen_class': data_info['unseen_class'][...],
                     'all_class':np.arange(nc),
                     'all_att': data_info['all_att'][...]}
    #load semantic data
    args.num_classes = nc
    args.sf_size = sf_size
    args.sf = semantic_data['all_att']
    
    traindir = os.path.join('./data',args.data_name,'train.list')
    valdir1 = os.path.join('./data',args.data_name,'test_seen.list')
    valdir2 = os.path.join('./data',args.data_name,'test_unseen.list')

    train_transforms, val_transforms = preprocess_strategy(args.data_name,args)

    train_dataset = datasets.ImageFolder(args.data,traindir,train_transforms)
    val_dataset1 = datasets.ImageFolder(args.data,valdir1, val_transforms)
    val_dataset2 = datasets.ImageFolder(args.data,valdir2, val_transforms)
    

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler1 = torch.utils.data.distributed.DistributedSampler(val_dataset1,shuffle=False)
        val_sampler2 = torch.utils.data.distributed.DistributedSampler(val_dataset2,shuffle=False)
        args.batch_size = int(args.batch_size/ngpus_per_node)
    else:
        train_sampler = None
        val_sampler1 = None
        val_sampler2 = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader1 = torch.utils.data.DataLoader(
        val_dataset1, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler1)
        
    val_loader2 = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data,valdir2, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler2)

    ''' model building '''
    model = models.BaseS2V(pretrained=True,args=args)

    if args.distributed:
        if args.is_syncbn:
            args.logger.info('Convert BN to SyncBN')
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
    else:
        model = torch.nn.DataParallel(model).cuda()
        
    criterions = [nn.CrossEntropyLoss().cuda(),
                 nn.MSELoss().cuda(),
                 ]

    ''' optimizer '''
    base_params = [v for k, v in model.named_parameters() if 'zsr' not in k]
    zsl_params = [v for k, v in model.named_parameters() if 'zsr' in k]
    train_params = [{'params': base_params, 'lr': 0.1*args.lr},
                    {'params': zsl_params, 'lr': args.lr},
                    ]
    optimizer = torch.optim.Adam(train_params,args.lr, betas=(0.5,0.999),weight_decay=args.weight_decay)
                     
    ''' optionally resume from a checkpoint'''
    if args.resume:
        if os.path.isfile(args.resume):
            args.logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            if(best_prec1==0):
                best_prec1 = checkpoint['best_prec1']
            args.logger.info('=> pretrained acc {:.4F}'.format(best_prec1))
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            args.logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            args.logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lr, args.epoch_decay)
        
        # train for one epoch
        train(train_loader,semantic_data, model, criterions, optimizer, epoch, args)
        
        # evaluate on validation set
        prec1 = validate(val_loader1, val_loader2, semantic_data, model, criterion, args)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save model
        save_path = os.path.join(args.save_path,args.model+('_{:.4f}.model').format(best_prec1))
        if is_best:
            if 1:#dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    #'optimizer' : optimizer.state_dict(),
                },filename=save_path)
                args.logger.info('saving!!!!')


def train(train_loader, semantic_data, model, criterions, optimizer, epoch, args):   
    Loss_s2v = AverageMeter('L_seg', ':.4f')
    Loss_cls = AverageMeter('loss_D', ':.4f')
    progress = ProgressMeter(len(train_loader),
        [Loss_s2v,Loss_cls], 
        prefix="Epoch: [{}]".format(epoch))
         
    # switch to train mode
    model.train()
    
    if(args.is_fix):
        freeze_bn(model) 

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        logit,logit_aux = model(input)
        # loss
        idx = torch.arange(logit.size(0)).long()
        L_s2v = (1-logit[idx,target]).mean()
        Loss_s2v.update(L_s2v)
        L_cls = criterions[0](logit_aux,target)
        Loss_cls.update(L_cls)
        total_loss = L_s2v + L_cls
        # update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if i % args.print_freq == 0:
            progress.display(i,args.logger)

def validate(val_loader1, val_loader2, semantic_data, model, criterion, args):

    ''' load semantic data'''
    seen_c = semantic_data['seen_class']
    unseen_c = semantic_data['unseen_class']
    all_c = semantic_data['all_class']
    
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader1):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            if args.flipping_test:                
                [N,M,C,H,W] = input.size()
                input = input.view(N*M,C,H,W)   #flipping test
            
            # inference
            logits,feats = model(input)
            
            
            if args.flipping_test: 
                odr_logit = F.softmax(odr_logit,dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
                zsl_logit = F.softmax(zsl_logit,dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
            else:
                odr_logit = odr_logit.cpu().numpy()
                zsl_logit = zsl_logit.cpu().numpy()
            zsl_logit_s = zsl_logit.copy();zsl_logit_s[:,unseen_c]=-1
            zsl_logit_t = zsl_logit.copy();zsl_logit_t[:,seen_c]=-1
            
			
            # evaluation
            if(i==0):
                gt_s = target.cpu().numpy()
                odr_pre_s = np.argmax(odr_logit, axis=1)
                if args.flipping_test: 
                    odr_prob_s = odr_logit
                else:
                    odr_prob_s = softmax(odr_logit)
                zsl_pre_sA = np.argmax(zsl_logit, axis=1)
                zsl_pre_sS = np.argmax(zsl_logit_s, axis=1)
                if args.flipping_test:
                    zsl_prob_s = zsl_logit_t
                else: 
                    zsl_prob_s = softmax(zsl_logit_t)
            else:
                gt_s = np.hstack([gt_s,target.cpu().numpy()])
                odr_pre_s = np.hstack([odr_pre_s,np.argmax(odr_logit, axis=1)])
                if args.flipping_test:
                    odr_prob_s = np.vstack([odr_prob_s,odr_logit])
                else:
                    odr_prob_s = np.vstack([odr_prob_s,softmax(odr_logit)])
                zsl_pre_sA = np.hstack([zsl_pre_sA,np.argmax(zsl_logit, axis=1)])
                zsl_pre_sS = np.hstack([zsl_pre_sS,np.argmax(zsl_logit_s, axis=1)])
                if args.flipping_test:
                    zsl_prob_s = np.vstack([zsl_prob_s,zsl_logit_t])
                else:
                    zsl_prob_s = np.vstack([zsl_prob_s,softmax(zsl_logit_t)])

        for i, (input, target) in enumerate(val_loader2):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            if args.flipping_test:                
                [N,M,C,H,W] = input.size()
                input = input.view(N*M,C,H,W)   #flipping test
                
            # inference
            logits,feats = model(input)         
            
            odr_logit = _get_concat_all_feat(logits[0])
            zsl_logit = _get_concat_all_feat(logits[1])
            target = _get_concat_all_feat(target)  
            
            if args.flipping_test: 
                odr_logit = F.softmax(odr_logit,dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
                zsl_logit = F.softmax(zsl_logit,dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
            else:
                odr_logit = odr_logit.cpu().numpy()
                zsl_logit = zsl_logit.cpu().numpy()
            zsl_logit_s = zsl_logit.copy();zsl_logit_s[:,unseen_c]=-1
            zsl_logit_t = zsl_logit.copy();zsl_logit_t[:,seen_c]=-1
                
            if(i==0):
                gt_t = target.cpu().numpy()
                odr_pre_t = np.argmax(odr_logit, axis=1)
                if args.flipping_test: 
                    odr_prob_t = odr_logit
                else:
                    odr_prob_t = softmax(odr_logit)
                zsl_pre_tA = np.argmax(zsl_logit, axis=1)
                zsl_pre_tT = np.argmax(zsl_logit_t, axis=1)
                if args.flipping_test: 
                    zsl_prob_t = zsl_logit_t
                else:
                    zsl_prob_t = softmax(zsl_logit_t)
            else:
                gt_t = np.hstack([gt_t,target.cpu().numpy()])
                odr_pre_t = np.hstack([odr_pre_t,np.argmax(odr_logit, axis=1)])
                if args.flipping_test: 
                    odr_prob_t = np.vstack([odr_prob_t,odr_logit])
                else:
                    odr_prob_t = np.vstack([odr_prob_t,softmax(odr_logit)])
                zsl_pre_tA = np.hstack([zsl_pre_tA,np.argmax(zsl_logit, axis=1)])
                zsl_pre_tT = np.hstack([zsl_pre_tT,np.argmax(zsl_logit_t, axis=1)])
                if args.flipping_test: 
                    zsl_prob_t = np.vstack([zsl_prob_t,zsl_logit_t])
                else:
                    zsl_prob_t = np.vstack([zsl_prob_t,softmax(zsl_logit_t)])
        
        
        odr_prob = np.vstack([odr_prob_s,odr_prob_t])
        zsl_prob = np.vstack([zsl_prob_s,zsl_prob_t])
        gt = np.hstack([gt_s,gt_t])
    
        SS = compute_class_accuracy_total(gt_s, zsl_pre_sS,seen_c)
        UU = compute_class_accuracy_total(gt_t, zsl_pre_tT,unseen_c)
        ST = compute_class_accuracy_total(gt_s, zsl_pre_sA,seen_c)
        UT = compute_class_accuracy_total(gt_t, zsl_pre_tA,unseen_c)
        H = 2*ST*UT/(ST+UT) 
        CLS = compute_class_accuracy_total(gt_s, odr_pre_s,seen_c)
        
        H_opt,S_opt,U_opt,Ds_opt,Du_opt,tau = post_process(odr_prob, zsl_prob, gt, gt_s.shape[0], seen_c,unseen_c, args.data)
        
        args.logger.info(' SS: {:.4f} UU: {:.4f} ST: {:.4f} UT: {:.4f} H: {:.4f}'
              .format(SS,UU,ST,UT,H))
        args.logger.info('CLS {:.4f} S_opt: {:.4f} U_opt {:.4f} H_opt {:.4f} Ds_opt {:.4f} Du_opt {:.4f} tau {:.4f}'
              .format(CLS, S_opt, U_opt,H_opt,Ds_opt,Du_opt,tau))
              
        H = max(H,H_opt)
    return H
    
def _get_concat_all_feat(tensor):
    all_feat = all_gather(tensor)
    all_feat[torch.distributed.get_rank()] = tensor
    output = torch.cat(all_feat, dim=0)
    return output

if __name__ == '__main__':
    main()
