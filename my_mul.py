import argparse
import builtins
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import timm
import torch

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
from torch.backends import cudnn
import torch.multiprocessing as mp
import torch.distributed as dist

from augly.image import (EncodingQuality, OneOf,
                         RandomBlur, RandomEmojiOverlay, RandomPixelization,
                         RandomRotation, ShufflePixels)
from augly.image.functional import overlay_emoji, overlay_image, overlay_text
from augly.image.transforms import BaseTransform
from augly.utils import pathmgr
from augly.utils.base_paths import MODULE_BASE_DIR
from augly.utils.constants import FONT_LIST_PATH, FONTS_DIR, SMILEY_EMOJI_DIR

from PIL import Image, ImageFilter
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed as pml_dist

from tqdm import tqdm

parser = argparse.ArgumentParser(description='training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--input-size', default=256, type=int)
parser.add_argument('--memory-size', default=10000, type=int)
parser.add_argument('--log', default='log.txt', type=str, metavar='PATH')


# 数据增强部分

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandomOverlayText(BaseTransform):
    def __init__(
            self,
            opacity: float = 1.0,
            p: float = 1.0,
    ):
        super().__init__(p)
        self.opacity = opacity

        with open(Path(FONTS_DIR) / FONT_LIST_PATH) as f:
            font_list = [s.strip() for s in f.readlines()]
            blacklist = [
                'TypeMyMusic',
                'PainttheSky-Regular',
            ]
            self.font_list = [
                f for f in font_list
                if all(_ not in f for _ in blacklist)
            ]

        self.font_lens = []
        for ff in self.font_list:
            font_file = Path(MODULE_BASE_DIR) / ff.replace('.ttf', '.pkl')
            with open(font_file, 'rb') as f:
                self.font_lens.append(len(pickle.load(f)))

    def apply_transform(
            self, image: Image.Image,
            metadata: Optional[List[Dict[str, Any]]] = None,
            bboxes: Optional[List[Tuple]] = None,
            bbox_format: Optional[str] = None,
    ) -> Image.Image:
        i = random.randrange(0, len(self.font_list))
        kwargs = dict(
            font_file=Path(MODULE_BASE_DIR) / self.font_list[i],
            font_size=random.uniform(0.1, 0.3),
            color=[random.randrange(0, 256) for _ in range(3)],
            x_pos=random.uniform(0.0, 0.5),
            metadata=metadata,
            opacity=self.opacity,
        )
        try:
            for j in range(random.randrange(1, 3)):
                if j == 0:
                    y_pos = random.uniform(0.0, 0.5)
                else:
                    y_pos += kwargs['font_size']
                image = overlay_text(
                    image,
                    text=[random.randrange(0, self.font_lens[i]) for _ in range(random.randrange(5, 10))],
                    y_pos=y_pos,
                    **kwargs,
                )
            return image
        except OSError:
            return image


class RandomOverlayImageAndResizedCrop(BaseTransform):
    def __init__(
            self,
            img_paths: List[Path],
            opacity_lower: float = 0.5,
            size_lower: float = 0.4,
            size_upper: float = 0.6,
            input_size: int = 224,
            moderate_scale_lower: float = 0.7,
            hard_scale_lower: float = 0.15,
            overlay_p: float = 0.05,
            p: float = 1.0,
    ):
        super().__init__(p)
        self.img_paths = img_paths
        self.opacity_lower = opacity_lower
        self.size_lower = size_lower
        self.size_upper = size_upper
        self.input_size = input_size
        self.moderate_scale_lower = moderate_scale_lower
        self.hard_scale_lower = hard_scale_lower
        self.overlay_p = overlay_p

    def apply_transform(
            self, image: Image.Image,
            metadata: Optional[List[Dict[str, Any]]] = None,
            bboxes: Optional[List[Tuple]] = None,
            bbox_format: Optional[str] = None,
    ) -> Image.Image:

        if random.uniform(0.0, 1.0) < self.overlay_p:
            if random.uniform(0.0, 1.0) > 0.5:
                background = Image.open(random.choice(self.img_paths))
                overlay = image
            else:
                background = image
                overlay = Image.open(random.choice(self.img_paths))

            overlay_size = random.uniform(self.size_lower, self.size_upper)
            image = overlay_image(
                background,
                overlay=overlay,
                opacity=random.uniform(self.opacity_lower, 1.0),
                overlay_size=overlay_size,
                x_pos=random.uniform(0.0, 1.0 - overlay_size),
                y_pos=random.uniform(0.0, 1.0 - overlay_size),
                metadata=metadata,
            )
            return transforms.RandomResizedCrop(self.input_size, scale=(self.moderate_scale_lower, 1.))(image)
        else:
            return transforms.RandomResizedCrop(self.input_size, scale=(self.hard_scale_lower, 1.))(image)


class RandomEmojiOverlay(BaseTransform):
    def __init__(
            self,
            emoji_directory: str = SMILEY_EMOJI_DIR,
            opacity: float = 1.0,
            p: float = 1.0,
    ):
        super().__init__(p)
        self.emoji_directory = emoji_directory
        self.emoji_paths = pathmgr.ls(emoji_directory)
        self.opacity = opacity

    def apply_transform(
            self, image: Image.Image,
            metadata: Optional[List[Dict[str, Any]]] = None,
            bboxes: Optional[List[Tuple]] = None,
            bbox_format: Optional[str] = None,
    ) -> Image.Image:
        emoji_path = random.choice(self.emoji_paths)
        return overlay_emoji(
            image,
            emoji_path=os.path.join(self.emoji_directory, emoji_path),
            opacity=self.opacity,
            emoji_size=random.uniform(0.1, 0.3),
            x_pos=random.uniform(0.0, 1.0),
            y_pos=random.uniform(0.0, 1.0),
            metadata=metadata,
        )


class RandomEdgeEnhance(BaseTransform):
    def __init__(
            self,
            mode=ImageFilter.EDGE_ENHANCE,
            p: float = 1.0,
    ):
        super().__init__(p)
        self.mode = mode

    def apply_transform(self, image: Image.Image, *args) -> Image.Image:
        return image.filter(self.mode)


class ShuffledAug:

    def __init__(self, aug_list):
        self.aug_list = aug_list

    def __call__(self, x):
        # without replacement
        shuffled_aug_list = random.sample(self.aug_list, len(self.aug_list))
        for op in shuffled_aug_list:
            x = op(x)
        return x


def convert2rgb(x):
    return x.convert('RGB')


# 模型相关
class Gem(nn.Module):

    def __init__(self, p=3.0, eps=1e-6):
        super(Gem, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)
        return x


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)


class MyNet(nn.Module):

    def __init__(self, backbone, out_dim=256, p=3.0):
        super(MyNet, self).__init__()

        self.backbone = backbone
        # 使用 gempool 替换 avgpool
        self.gem = Gem(p=p)
        self.fc = nn.Linear(self.backbone.feature_info.info[-1]['num_chs'], out_dim, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_dim)

        # 初始化
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        self.norm = L2Norm()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)[-1]
        x = self.gem(x).view(batch_size, -1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.norm(x)
        return x


# 自定义数据集产生
class MyDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            paths,
            transforms,
    ):
        self.paths = paths
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i])
        image = self.transforms(image)
        return i, image


# 利用图像增强生成对比学习图像对
class NCropsTransform:
    """Take n random crops of one image as the query and key."""

    def __init__(self, aug_moderate, aug_hard, ncrops=2):
        self.aug_moderate = aug_moderate
        self.aug_hard = aug_hard
        self.ncrops = ncrops

    def __call__(self, x):
        return [self.aug_moderate(x)] + [self.aug_hard(x) for _ in range(self.ncrops - 1)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.distributed:

        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier(device_ids=[args.gpu])

    backbone = timm.create_model('mobilenetv3_small_050', features_only=True, pretrained=True)
    model = MyNet(backbone)

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])


    # 损失函数
    loss_fn = losses.ContrastiveLoss(pos_margin=0.0, neg_margin=1.0)
    # loss_fn = losses.NTXentLoss(temperature=0.5)
    loss_fn = losses.CrossBatchMemory(loss_fn, embedding_size=256, memory_size=args.memory_size)
    loss_fn = pml_dist.DistributedLossWrapper(loss=loss_fn)

    init_lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    # 数据集目录
    data_path = list(Path(args.data).glob('**/*.jpg'))

    norm = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # 复合数据增强
    aug_moderate = [
        transforms.RandomResizedCrop(args.input_size, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm
    ]
    aug_list = [
        transforms.ColorJitter(0.7, 0.7, 0.7, 0.2),
        RandomPixelization(p=0.25),
        ShufflePixels(factor=0.1, p=0.25),
        OneOf([EncodingQuality(quality=q) for q in [10, 20, 30, 50]], p=0.25),
        transforms.RandomGrayscale(p=0.25),
        RandomBlur(p=0.25),
        transforms.RandomPerspective(p=0.25),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        RandomOverlayText(p=0.25),
        RandomEmojiOverlay(p=0.25),
        OneOf([RandomEdgeEnhance(mode=ImageFilter.EDGE_ENHANCE), RandomEdgeEnhance(mode=ImageFilter.EDGE_ENHANCE_MORE)],
              p=0.25),
    ]
    aug_hard = [
        RandomRotation(p=0.25),
        RandomOverlayImageAndResizedCrop(
            data_path, opacity_lower=0.6, size_lower=0.4, size_upper=0.6,
            input_size=args.input_size, moderate_scale_lower=0.7, hard_scale_lower=0.15, overlay_p=0.05, p=1.0,
        ),
        ShuffledAug(aug_list),
        convert2rgb,
        transforms.ToTensor(),
        transforms.RandomErasing(value='random', p=0.25),
        norm
    ]

    train_dataset = MyDataset(
        data_path,
        NCropsTransform(
            transforms.Compose(aug_moderate),
            transforms.Compose(aug_hard),
            ncrops=2
        )
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)


    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        train_one_epoch(train_loader, model, loss_fn, optimizer, scaler, epoch, args)

        if args.multiprocessing_distributed and args.rank % ngpus_per_node == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }, filename=f'/train/checkpoint_{epoch:04d}.pth.tar')

    torch.save(model.state_dict(), "model.pth")


def train_one_epoch(train_loader, model, loss_fn, optimizer, scaler, epoch, args):
    losses = AverageMeter('Loss', ':.3f')
    progress = tqdm(train_loader, desc=f'epoch {epoch}', leave=False, total=len(train_loader))

    model.train()

    for labels, images in progress:
        optimizer.zero_grad()

        labels = labels.cuda(args.gpu, non_blocking=True)
        # 重复标签
        labels = torch.tile(labels, dims=(2,))
        # 数组增强后会得到一对 Key 和 Query，展平Batch
        images = torch.cat([
            image for image in images
        ], dim=0).cuda(args.gpu, non_blocking=True)

        with torch.cuda.amp.autocast():
            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

        losses.update(loss.item(), images.size(0))

        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress.set_postfix(loss=losses.avg)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)



def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * (1 - (epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
