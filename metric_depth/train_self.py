import argparse
import logging
import os
import pprint
import random
import time

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.c3vd import C3VD, C3VD_flip_and_swap
from dataset.UCL import *
from dataset.SimCol import SimCol, SimCol_flip_and_swap
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log


# # sleep 4 小时
# import time
# print("sleeping")
# time.sleep(4*60*60)


parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='UCL_self', choices=['hypersim', 'vkitti', 'UCL', 'UCL_flip_and_swap', 'UCL_pps', 'UCL_aug', 'UCL_self', 'c3vd', 'c3vd_flip_and_swap', 'SimCol', 'SimCol_flip_and_swap'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=200, type=float) # UCL SimCol 200mm, C3VD 100mm
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--bs', default=2, type=int) # batch_size
parser.add_argument('--lr', default=0.000005, type=float)
# parser.add_argument('--pretrained-from', default='/Disk_2/ZanXin/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', type=str)
parser.add_argument('--pretrained-from', default='/home/jiahan/jiahan/codes/Depth-Anything-V2/tmp/2024-10-07_14-17-16/latest_4.pth', type=str)
parser.add_argument('--init-from', default='/home/jiahan/jiahan/codes/Depth-Anything-V2/tmp/2024-10-08_21-14-04/latest_4.pth', type=str)
parser.add_argument('--is-infer', action='store_true')
parser.add_argument('--save-path', default='./tmp',type=str, required=False)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--frozen-encoder', default=False, type=bool)


def main():
    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path, time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))
    os.makedirs(args.save_path, exist_ok=True)
    
    warnings.simplefilter('ignore', np.RankWarning) # 忽略 Numpy 中的秩警告
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    print("!!! ", args.is_infer)
    # rank, world_size = setup_distributed(port=args.port) # 设置分布式训练环境，包括获取当前进程的 rank 或者 world_size(num_gpus)
    
    rank = 0
    world_size = 1
    
    if rank == 0: # 主进程
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path) # 训练过程中记录日志和可视化
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
   # dist.init_process_group(backend='nccl')
   
   # 训练之前推理
    print("========== infering ===========")
    # os.system(f"python scripts/run_ucl_metric.py --load-from {args.pretrained_from}")
    os.system(f"python scripts/run_ucl_metric.py --load-from {args.init_from}")
    print("===============================")
    
    size = (args.img_size, args.img_size)
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size) # train datasets
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
    elif args.dataset == 'c3vd':  # if c3vd
        trainset = C3VD('metric_depth/dataset/splits/c3vd/train.txt', 'train', size=size)
    elif args.dataset == "c3vd_flip_and_swap":
        trainset = C3VD_flip_and_swap('metric_depth/dataset/splits/c3vd/train.txt', 'train', size=size)
    elif args.dataset == 'UCL':
        trainset = UCL('metric_depth/dataset/splits/UCL_aug/train.txt', 'train', size=size)
    elif args.dataset == 'UCL_flip_and_swap':
        trainset = UCL_flip_and_swap('metric_depth/dataset/splits/UCL/train.txt', 'train', size=size)
    elif args.dataset == 'UCL_aug':
        trainset = UCL_aug('metric_depth/dataset/splits/UCL_aug/train.txt', 'train', size=size)
    elif args.dataset == 'UCL_self':
        trainset = UCL_self('metric_depth/dataset/splits/UCL_self/train.txt', 'train', size=size)
    elif args.dataset == 'SimCol':
        trainset = SimCol('metric_depth/dataset/splits/Simcol/train.txt', 'train', size=size)
    elif args.dataset == 'SimCol_flip_and_swap':
        trainset = SimCol_flip_and_swap('metric_depth/dataset/splits/Simcol/train.txt', 'train', size=size)
    else:
        raise NotImplementedError
    # trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # trainsampler = torch.utils.data.SequentialSampler(trainset) # zx
    trainsampler = RandomSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler) # zx num_worker
    
    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size) # validation datasets
    elif args.dataset == 'vkitti':
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    elif args.dataset == 'c3vd':
        valset = C3VD('metric_depth/dataset/splits/c3vd/val.txt', 'val', size=size)
    elif args.dataset == 'c3vd_flip_and_swap':
        valset = C3VD_flip_and_swap('metric_depth/dataset/splits/c3vd/val.txt', 'val', size=size)
    elif args.dataset == 'UCL':
        valset = UCL('metric_depth/dataset/splits/UCL_aug/val.txt', 'val', size=size)
    elif args.dataset == 'UCL_flip_and_swap':
        valset = UCL_flip_and_swap('metric_depth/dataset/splits/UCL/val.txt', 'val', size=size)
    elif args.dataset == 'UCL_aug':
        valset = UCL_aug('metric_depth/dataset/splits/UCL_aug/val.txt', 'val', size=size)
    elif args.dataset == 'UCL_self':
        valset = UCL('metric_depth/dataset/splits/UCL_self/val.txt', 'val', size=size)
    elif args.dataset == 'SimCol':
        valset = SimCol('metric_depth/dataset/splits/Simcol/val.txt', 'val', size=size)
    elif args.dataset == 'SimCol_flip_and_swap':
        valset = SimCol_flip_and_swap('metric_depth/dataset/splits/Simcol/val.txt', 'val', size=size)
    else:
        raise NotImplementedError
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valsampler = torch.utils.data.SequentialSampler(valset) #zx
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
    
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = 0
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        # 'metric_hypersim_vitl': {'encoder': 'metric_hypersim_vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    if args.frozen_encoder: model.pretrained.requires_grad_(False) # ! 冻结encoder
    
    if args.pretrained_from:
        weight = torch.load(args.pretrained_from, map_location='cpu')
        if 'model' in weight.keys(): weight = weight['model']
        else: pass
        model.load_state_dict(weight)
        
        # ! NOTE 下面的代码有BUG，会加载失败，这是因为我们保存的dict没有这个键，但是strict是False，导致不会出现报错
        # model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # 保持 BatchNorm 层的统计数据同步
    
    # model = torch.nn.DataParallel(model) # ! #jiahan 不启用多个GPU训练
    
    model.cuda(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
    #                                                   output_device=local_rank, find_unused_parameters=True) # 实现模型并行化
    
    criterion = SiLogLoss().cuda(local_rank)
    
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    total_iters = args.epochs * len(trainloader)
    
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    
    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                            epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
                            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))
        
        # trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()
            
            if random.random() < 0.5: # 提高模型的鲁棒性和泛化能力
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)
            
            pred = model(img) # img -> prediction depth img
            
            loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            # print(i)
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
            
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)
                writer.add_scalar('train/MAE', (pred - depth).abs().mean(), iters)
            
            if rank == 0 and i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))


        model.eval()
        
        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        nsamples = torch.tensor([0.0]).cuda()
        
        for i, sample in enumerate(valloader):
            
            img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
            
            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
            valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            
            if valid_mask.sum() < 10:
                continue
            
            # pred = pred.unsqueeze(1)  # 在第二个维度上增加一个维度, 变成 [256, 1, 3]
            # pred = pred.expand(-1, 256, -1)  # 在第二个维度上复制 256 次, 变成 [256, 256, 3]

            cur_results = eval_depth(pred[valid_mask], depth[valid_mask]) # pred : [256, 3] -> [256, 256, 3]
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1

            # # print(i)
            # if i >= 8000:
            #     break
        
        # torch.distributed.barrier()
        
        # for k in results.keys():
        #     dist.reduce(results[k], dst=0)
        # dist.reduce(nsamples, dst=0)
        
        if rank == 0:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
            logger.info('==========================================================================================')
            print()
            
            for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, f'latest_{epoch}.pth'))

        print("finish this epoch!!!!!")
        
        if args.is_infer:
            print("========== infering ===========")
            os.system(f"python scripts/run_ucl_metric.py --load-from {os.path.join(args.save_path, f'latest_{epoch}.pth')}")
            print("===============================")


if __name__ == '__main__':
    main()