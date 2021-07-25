
from __future__ import absolute_import, division, print_function
import os

import torch
import torch.nn as nn
from torch.utils import data

import argparse
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from libs.datasets import get_transforms, get_datasets
from libs.networks import VideoModel
from libs.networks.loss import KLDLoss
from libs.utils.metric import StructureMeasure
from libs.utils.pyt_utils import load_model
from libs.utils.logger import get_logger
from libs.utils.utils import adjust_lr, clip_gradient, adjust_lr_finetune
from libs.utils import pytorch_iou, pytorch_ssim

configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        spshot=20000,
        nclass=2,
        sshow=10,
    ),
    'stage2_cfg': dict(
        NUM_BRANCHES = 2,
        NUM_CHANNELS = [32, 64],
        NUM_BLOCKS = [4, 4],
    ),
    'stage3_cfg': dict(
        NUM_BRANCHES = 3,
        NUM_CHANNELS=[256, 512, 1024],
        NUM_BLOCKS=[4, 4, 4],
    ),
    'stage4_cfg': dict(
        NUM_BRANCHES = 4,
        NUM_BLOCKS = [4, 4, 4, 4],
        NUM_CHANNELS = [256, 512, 1024, 256],
    )
}

CFG = configurations
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='/media/lewis/Win 10 Pro x64/datasets/',
                    help='path to datasets folder')
parser.add_argument('--checkpoint', default='models/image_epoch.pth',
                    help='path to the pretrained checkpoint')
parser.add_argument('--dataset-config', default='config/datasets.yaml',
                    help='dataset config file')
parser.add_argument('--save-folder', default='models/checkpoints',
                    help='location to save checkpoint models')
parser.add_argument('-j', '--num_workers', default=1, type=int, metavar='N',
                    help='number of data loading workers.')
parser.add_argument('--log_dir', default='config/',
                    help='log_dir file')
parser.add_argument('--log_file', default='config/',
                    help='log_file file')


parser.add_argument('--batch-size', default=1, type=int,
                    help='batch size for each gpu.')
parser.add_argument('--backup-epochs', type=int, default=1,
                    help='iteration epoch to perform state backups')
parser.add_argument('--epochs', type=int, default=150,
                    help='upper epoch limit')
parser.add_argument('--start-epoch', type=int, default=0,
                    help='epoch number to resume')
parser.add_argument('--eval-first', default=False, action='store_true',
                    help='evaluate model weights before training')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                    help='initial learning rate')

parser.add_argument('--lr_mode', type=str, default="poly")
parser.add_argument('--base_lr', type=float, default=1e-5)
parser.add_argument('--finetune_lr', type=float, default=1e-6)
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--clip_margin', type=float, default=0.5, help='gradient clipping margin')

parser.add_argument('--size', default=448, type=int,
                    help='image size')
parser.add_argument('--os', default=16, type=int,
                    help='output stride.')
parser.add_argument("--clip_len", type=int, default=4,
                    help="the number of frames in a video clip.")

args = parser.parse_args()


def KLDivLoss(Stu_output, Tea_output, temperature = 1):
    T = temperature
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(Stu_output/T, dim=1), F.softmax(Tea_output/T, dim=1))
    KD_loss = KD_loss * T * T
    return KD_loss


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

if cuda:
    torch.backends.cudnn.benchmark = True
    current_device = torch.cuda.current_device()
    print("Running on", torch.cuda.get_device_name(current_device))
else:
    print("Running on CPU")
data_transforms = get_transforms(
    input_size=(args.size, args.size),
    image_mode=True
)


train_dataset = get_datasets(
    name_list=["DAVIS2016"],
    split_list=["train"],
    config_path=args.dataset_config,
    root=args.data,
    training=True,
    transforms=data_transforms['train'],
    read_clip=True,
    random_reverse_clip=False,
    clip_len=args.clip_len
)

val_dataset = get_datasets(
    name_list=["DAVIS2016"],
    split_list=["val"],
    config_path=args.dataset_config,
    root=args.data,
    training=True,
    transforms=data_transforms['val'],
    read_clip=True,
    random_reverse_clip=False,
    clip_len=args.clip_len
)

train_dataloader = data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    drop_last=True
)
val_dataloader = data.DataLoader(
    dataset=val_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False
)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}


model = VideoModel(output_stride=args.os, pretrained=True, cfg=CFG)
# # load pretrained models
if os.path.exists(args.checkpoint):
    print('Loading state dict from: {0}'.format(args.checkpoint))
    logger = get_logger()
    if args.start_epoch == 0:
        # load_backbone(model, args.checkpoint, logger)
        model = load_model(model=model, model_file=args.checkpoint)
    else:
        # load_backbone(model, args.checkpoint, logger)
        model = load_model(model=model, model_file=args.checkpoint)
else:
    raise ValueError("Cannot find model file at {}".format(args.checkpoint))


model = nn.DataParallel(model)
model.to(device)
# print(model)


bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def bce_ssim_loss(pred, target):

    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + ssim_out + iou_out

    return loss


def muti_bce_loss_fusion(pred, block4_s, bu1_s, bu2_s, bu3_s, labels):

    loss0 = bce_ssim_loss(pred, labels)
    loss1 = bce_ssim_loss(block4_s, labels)
    loss2 = bce_ssim_loss(bu1_s, labels)
    loss3 = bce_ssim_loss(bu2_s, labels)
    loss4 = bce_ssim_loss(bu3_s, labels)

    loss = loss0 + loss1 + loss2 + loss3 + loss4
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f\n" %
    #       (loss0.data, loss1.data, loss2.data, loss3.data, loss4.data))

    return loss


criterion_bce = nn.BCEWithLogitsLoss()
criterion_KL = KLDLoss()

finetune_params = [params for name, params in model.named_parameters() if ("backbone" in name)]
base_params = [params for name, params in model.named_parameters() if ("backbone" not in name)]

optimizer = torch.optim.Adam([
    {'params': base_params, 'lr': args.base_lr, 'weight_decay': 1e-4, 'name': "base_params"},
    {'params': finetune_params, 'lr': args.finetune_lr, 'weight_decay': 1e-4, 'name': 'finetune_params'}])
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


def train():

    best_smeasure = 0.0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs+1):

        # print('base_params lr: {:.8f}   finetune_params lr: {:.8f}'.format(optimizer.state_dict()['param_groups'][0]['lr'], optimizer.state_dict()['param_groups'][1]['lr']))
        # print('finetune_params lr: {:.8f}'.format(optimizer.state_dict()['param_groups'][1]['lr']))

        if args.eval_first:
            phases = ['val']
        else:
            phases = ['train', 'val']

        for phase in phases:
            if phase == 'train':
                model.train()
                # model.freeze_bn()
                model.module.freeze_bn()
            else:
                model.eval()

            running_loss = 0.0
            running_mae = 0.0
            running_smean = 0.0

            cur_epoch = epoch
            total_epoch = args.epochs
            batches_per_epoch = len(dataloaders[phase])
            cur_batches = 0

            # alpha = 0.01
            print("{} epoch {}...".format(phase, epoch))
            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                cur_batches = cur_batches + 1
                # is_fixation = 0
                images, labels = [], []
                for frame in data:
                    images.append(frame['image'].to(device))
                    labels.append(frame['label'].to(device))
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # read clips
                    preds = model(images)
                    loss = []
                    for pred, label in zip(preds, labels):
                        loss.append(bce_ssim_loss(pred, label))
                        # loss.append(criterion_bce(block4_affinity_supervision, label))

                    if phase == 'train':
                        # base_lr, finetune_lr = adjust_learning_rate(cur_epoch, cur_batches, batches_per_epoch, total_epoch)
                        base_lr, finetune_lr = adjust_lr_finetune(optimizer, args.base_lr, args.finetune_lr, epoch, args.decay_rate, args.decay_epoch)
                        torch.autograd.backward(loss)
                        optimizer.step()

                for _loss in loss:
                    running_loss += _loss.item()

                for i, (label_, pred_) in enumerate(zip(labels, preds)):
                    for j, (label, pred) in enumerate(zip(label_.detach().cpu(), pred_.detach().cpu())):
                        pred_idx = pred[0,:,:].numpy()
                        label_idx = label[0,:,:].numpy()
                        if phase == 'val':
                            running_smean += StructureMeasure(pred_idx.astype(np.float32), (label_idx>=0.5).astype(np.bool))
                        running_mae += np.abs(pred_idx - label_idx).mean()

            samples_num = len(dataloaders[phase].dataset)
            samples_num *= args.clip_len

            epoch_loss = running_loss / samples_num
            epoch_mae = running_mae / samples_num
            # print{'\n'}
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print('{} MAE: {:.4f}'.format(phase, epoch_mae))

            # save current best epoch
            if phase == 'val':
                epoch_smeasure = running_smean / samples_num
                print('{} S-measure: {:.4f}'.format(phase, epoch_smeasure))
                if epoch_smeasure > best_smeasure:
                    best_smeasure = epoch_smeasure
                    best_epoch = epoch
                    # model_path = os.path.join(args.save_folder, "video_current_best_model_" + str(best_epoch) + ".pth")
                    model_path = os.path.join(args.save_folder, "video_current_best_model.pth")
                    print("Saving current best model at: {}".format(model_path) )
                    torch.save(
                        # model.state_dict(),
                        model.module.state_dict(),
                        model_path,
                    )
        if epoch > 0 and epoch % args.backup_epochs == 0:
            # save model
            model_path = os.path.join(args.save_folder, "video_epoch-{}.pth".format(epoch))
            print("Backup model at: {}".format(model_path))
            torch.save(
                # model.state_dict(),
                model.module.state_dict(),
                model_path,
            )

    print('Best S-measure: {} at epoch {}'.format(best_smeasure, best_epoch))


if __name__ == "__main__":
    train()
