# System libs
import os
import time
# import math
import random
import argparse
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.io import loadmat
from scipy.misc import imresize, imsave
# Our libs
from dataset import CityScapes, trainID2Class
from models import ModelBuilder, NovelViewHomography
from utils import AverageMeter, colorEncode, accuracy, make_variable, intersectionAndUnion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def forward_with_loss(nets, batch_data, use_seg_label=True):
    (net_encoder, net_decoder_1, net_decoder_2, warp, crit1, crit2) = nets
    
    (imgs, segs, view2, intrinsics, baseline, depth, normals, infos) = batch_data
    
    input_img = Variable(imgs)
    input_img = input_img.cuda()
    
    # feed input data
    if use_seg_label: # supervised training
        
        label_seg = Variable(segs)
        label_seg = label_seg.cuda()
        
        # forward
        featuremap = net_encoder(input_img)
        seg_mask = net_decoder_1(featuremap)
        
        err = crit1(seg_mask, label_seg)
        
        return seg_mask, err
        
    else: # unsupervised training
        
        view2_img = Variable(view2)
        view2_img = view2_img.cuda()
        
        intrs = Variable(intrinsics)
        intrs = intrs.cuda()
        
        # forward
        featuremap = net_encoder(input_img)
        seg_mask = net_decoder_1(featuremap)
        img_recon = warp(net_decoder_2(featuremap, seg_mask))
        
        err = crit2(img_recon, view2_img)
        
        return seg_mask, err


def visualize(batch_data, pred, args):
    colors = loadmat('./colormap.mat')['colors']
    (imgs, segs, view2, intrinsics, baseline, depth, normals, infos) = batch_data
    for j in range(len(infos)):
        # get/recover image
        # img = imread(os.path.join(args.root_img, infos[j]))
        img = imgs[j].clone()
        for t, m, s in zip(img,
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

        # segmentation
        lab = segs[j].numpy()
        lab_color = colorEncode(lab, colors)

        # prediction
        pred_ = np.argmax(pred.data.cpu()[j].numpy(), axis=0)
        pred_color = colorEncode(pred_, colors)

        # aggregate images and save
        im_vis = np.concatenate((img, lab_color, pred_color),
                                axis=1).astype(np.uint8)
        imsave(os.path.join(args.vis,
                            infos[j].replace('/', '_')), im_vis)


# train one epoch
def train(nets, loader_sup, loader_unsup, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    for net in nets:
        if not args.fix_bn:
            net.train()
        else:
            net.eval()

    # main supervised loop
    for iteration in range(args.gamma):
        tic = time.time()
        for i, batch_data in enumerate(loader_sup):
            i += iteration * len(loader_sup)
            data_time.update(time.time() - tic)
            for net in nets:
                net.zero_grad()

            # forward pass
            pred, err = forward_with_loss(nets, batch_data, use_seg_label=True)
            err = args.beta * err
            
            # Backward
            err.backward()

            for optimizer in optimizers:
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # calculate accuracy, and display
            if i % args.disp_iter == 0:
                acc, _ = accuracy(batch_data, pred)

                print('Epoch: [{}][{}/{}][Sup], Time: {:.2f}, Data: {:.2f}, '
                    'lr_encoder: {}, lr_decoder: {}, '
                    'Accuracy: {:4.2f}%, Loss: {}'
                    .format(epoch, i, args.epoch_iters,
                            batch_time.average(), data_time.average(),
                            args.lr_encoder, args.lr_decoder,
                            acc * 100, err.data.item()))

                fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
                history['train']['epoch'].append(fractional_epoch)
                history['train']['err'].append(err.data.item())
                history['train']['acc'].append(acc)

    # main unsupervised loop
    tic = time.time()
    for i, batch_data in enumerate(loader_unsup):
        i += args.gamma * len(loader_sup)
        data_time.update(time.time() - tic)
        for net in nets:
            net.zero_grad()

        # forward pass
        pred, err = forward_with_loss(nets, batch_data, use_seg_label=False)

        # Backward
        err.backward()

        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # calculate accuracy, and display
        if i % args.disp_iter == 0:
            acc, _ = accuracy(batch_data, pred)

            print('Epoch: [{}][{}/{}][Unsup], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {}, lr_decoder: {}, '
                  'Accuracy: {:4.2f}%, Loss: {}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_encoder, args.lr_decoder,
                          acc * 100, err.data.item()))
            
def evaluate(nets, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    # switch to eval mode
    for net in nets:
        net.eval()

    for i, batch_data in enumerate(loader):
        # forward pass
        pred, err = forward_with_loss(nets, batch_data, use_seg_label=True)
        loss_meter.update(err.data.item())
        print('[Eval] iter {}, loss: {}'.format(i, err.data.item()))

        # calculate accuracy
        acc, pix = accuracy(batch_data, pred)
        acc_meter.update(acc, pix)

        intersection, union = intersectionAndUnion(batch_data, pred,
                                                   args.num_class)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # visualization
        visualize(batch_data, pred, args)

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(trainID2Class[i], _iou))

    print('[Eval Summary]:')
    print('Epoch: {}, Loss: {}, Mean IoU: {:.4}, Accurarcy: {:.2f}%'
          .format(epoch, loss_meter.average(), iou.mean(), acc_meter.average() * 100))

    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['acc'].append(acc_meter.average())
    history['val']['mIoU'].append(iou.mean())

    # Plot figure
    if epoch > 0:
        print('Plotting loss figure...')
        fig = plt.figure()
        plt.plot(np.asarray(history['train']['epoch']),
                 np.log(np.asarray(history['train']['err'])),
                 color='b', label='training')
        plt.plot(np.asarray(history['val']['epoch']),
                 np.log(np.asarray(history['val']['err'])),
                 color='c', label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Log(loss)')
        fig.savefig('{}/loss.png'.format(args.ckpt), dpi=200)
        plt.close('all')

        fig = plt.figure()
        plt.plot(history['train']['epoch'], history['train']['acc'],
                 color='b', label='training')
        plt.plot(history['val']['epoch'], history['val']['acc'],
                 color='c', label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        fig.savefig('{}/accuracy.png'.format(args.ckpt), dpi=200)
        plt.close('all')


def checkpoint(nets, history, args):
    print('Saving checkpoints...')
    (net_encoder, net_decoder_1, net_decoder_2, warp, crit1, crit2) = nets
    suffix_latest = 'latest.pth'
    suffix_best_acc = 'best_acc.pth'
    suffix_best_mIoU = 'best_mIoU.pth'
    suffix_best_err = 'best_err.pth'

    if args.num_gpus > 1:
        dict_encoder = net_encoder.module.state_dict()
        dict_decoder_1 = net_decoder_1.module.state_dict()
        dict_decoder_2 = net_decoder_2.module.state_dict()

    else:
        dict_encoder = net_encoder.state_dict()
        dict_decoder_1 = net_decoder_1.state_dict()
        dict_decoder_2 = net_decoder_2.state_dict()


    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_encoder,
               '{}/encoder_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_decoder_1,
               '{}/decoder_1_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_decoder_2,
               '{}/decoder_2_{}'.format(args.ckpt, suffix_latest))

    cur_err = history['val']['err'][-1]
    cur_acc = history['val']['acc'][-1]
    cur_mIoU = history['val']['mIoU'][-1]

    if cur_acc > args.best_acc:
        # save best accuracy instead
        # args.best_err = cur_err
        args.best_acc = cur_acc
        torch.save(history,
                   '{}/history_{}'.format(args.ckpt, suffix_best_acc))
        torch.save(dict_encoder,
                   '{}/encoder_{}'.format(args.ckpt, suffix_best_acc))
        torch.save(dict_decoder_1,
                   '{}/decoder_1_{}'.format(args.ckpt, suffix_best_acc))
        torch.save(dict_decoder_2,
                   '{}/decoder_2_{}'.format(args.ckpt, suffix_best_acc))

    if cur_mIoU > args.best_mIoU:
        # save best accuracy instead
        # args.best_err = cur_err
        args.best_mIoU = cur_mIoU
        torch.save(history,
                   '{}/history_{}'.format(args.ckpt, suffix_best_mIoU))
        torch.save(dict_encoder,
                   '{}/encoder_{}'.format(args.ckpt, suffix_best_mIoU))
        torch.save(dict_decoder_1,
                   '{}/decoder_1_{}'.format(args.ckpt, suffix_best_mIoU))
        torch.save(dict_decoder_2,
                   '{}/decoder_2_{}'.format(args.ckpt, suffix_best_mIoU))

    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(history,
                   '{}/history_{}'.format(args.ckpt, suffix_best_err))
        torch.save(dict_encoder,
                   '{}/encoder_{}'.format(args.ckpt, suffix_best_err))
        torch.save(dict_decoder_1,
                   '{}/decoder_1_{}'.format(args.ckpt, suffix_best_err))
        torch.save(dict_decoder_2,
                   '{}/decoder_2_{}'.format(args.ckpt, suffix_best_err))


def create_optimizers(nets, args):
    (net_encoder, net_decoder_1, net_decoder_2, warp, crit1, crit2) = nets
    optimizer_encoder = torch.optim.SGD(
        net_encoder.parameters(),
        lr=args.lr_encoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    optimizer_decoder_1 = torch.optim.SGD(
        net_decoder_1.parameters(),
        lr=args.lr_decoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    optimizer_decoder_2 = torch.optim.SGD(
        net_decoder_2.parameters(),
        lr=args.lr_decoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)

    return (optimizer_encoder, optimizer_decoder_1, optimizer_decoder_2)


def adjust_learning_rate(optimizers, epoch, args):
    drop_ratio = (1. * (args.num_epoch - epoch) / (args.num_epoch - epoch + 1)) \
                 ** args.lr_pow
    args.lr_encoder *= drop_ratio
    args.lr_decoder *= drop_ratio
    (optimizer_encoder, optimizer_decoder_1, optimizer_decoder_2) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = args.lr_encoder
    for param_group in optimizer_decoder_1.param_groups:
        param_group['lr'] = args.lr_decoder
    for param_group in optimizer_decoder_2.param_groups:
        param_group['lr'] = args.lr_decoder


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(weights=args.weights_encoder)
    net_decoder_1 = builder.build_decoder(weights=args.weights_decoder)
    net_decoder_2 = builder.build_decoder(arch='c1',num_class=args.num_class,
                                          num_plane=args.num_plane, use_softmax=False,
                                          weights=args.weights_plane_net)
    
    # Warp application module
    warp = NovelViewHomography()
    
    if args.weighted_class:
        crit1 = nn.NLLLoss(ignore_index=-1, weight=args.class_weight)
    else:
        crit1 = nn.NLLLoss(ignore_index=-1)
    crit2 = nn.MSELoss()

    # Dataset and Loader
    dataset_train_sup = CityScapes('train', root=args.root_cityscapes, cropSize=args.imgSize,
                             max_sample=args.num_sup, is_train=1)
    dataset_train_unsup = CityScapes('train', root=args.root_cityscapes, cropSize=args.imgSize,
                             max_sample=-1, is_train=1)
    dataset_val = CityScapes('val', root=args.root_cityscapes, cropSize=args.imgSize,
                             max_sample=args.num_val, is_train=0)

    loader_train_sup = torch.utils.data.DataLoader(
        dataset_train_sup,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_train_unsup = torch.utils.data.DataLoader(
        dataset_train_unsup,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=True)

    args.epoch_iters = int((args.gamma * len(dataset_train_sup) + len(dataset_train_unsup)) / args.batch_size)
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # load nets into gpu
    if args.num_gpus > 1:
        net_encoder = nn.DataParallel(net_encoder,
                                      device_ids=range(args.num_gpus))
        net_decoder_1 = nn.DataParallel(net_decoder_1,
                                        device_ids=range(args.num_gpus))
        net_decoder_2 = nn.DataParallel(net_decoder_2,
                                        device_ids=range(args.num_gpus))

    nets = (net_encoder, net_decoder_1, net_decoder_2, warp, crit1, crit2)
    for net in nets:
        net.cuda()

    # Set up optimizers
    optimizers = create_optimizers(nets, args)

    # Main loop
    history = {split: {'epoch': [], 'err': [], 'acc': [], 'mIoU': []}
               for split in ('train', 'val')}

    # optional initial eval
    evaluate(nets, loader_val, history, 0, args)
    for epoch in range(1, args.num_epoch + 1):
        train(nets, loader_train_sup, loader_train_unsup, optimizers, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(nets, loader_val, history, epoch, args)

        # checkpointing
        checkpoint(nets, history, args)

        # adjust learning rate
        adjust_learning_rate(optimizers, epoch, args)

    print('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='sup_only',
                        help="a name for identifying the experiment")
    parser.add_argument('--weights_encoder',
                        default='/home/selfdriving/kchitta/Style-Randomization/pretrained/encoder_cityscapes.pth',
                        help="weights to initialize encoder")
    parser.add_argument('--weights_decoder',
                        default='/home/selfdriving/kchitta/Style-Randomization/pretrained/decoder_1_cityscapes.pth',
                        help="weights to initialize segmentation branch")
    parser.add_argument('--weights_plane_net',
                        default='',
                        help="weights to initialize reconstruction branch")

    # Path related arguments
    parser.add_argument('--root_cityscapes',
                        default='/home/selfdriving/datasets/cityscapes_full')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=3, type=int,
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu', default=3, type=int,
                        help='input batch size')
    parser.add_argument('--batch_size_per_gpu_eval', default=1, type=int,
                        help='eval batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')

    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_encoder', default=1e-3, type=float, help='LR')
    parser.add_argument('--lr_decoder', default=1e-3, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--gamma', default=5, type=float,
                        help='number of repetitions of supervised training per epoch')
    parser.add_argument('--beta', default=1, type=float,
                        help='relative weight of the supervised loss')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--fix_bn', default=0, type=int,
                        help='fix bn params')

    # Data related arguments
    parser.add_argument('--num_sup', default=100, type=int,
                        help='number of images for supervised training')
    parser.add_argument('--num_val', default=50, type=int,
                        help='number of images to evaluate')
    parser.add_argument('--num_class', default=19, type=int,
                        help='number of classes')
    parser.add_argument('--num_plane', default=3, type=int,
                        help='number of planes')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=720, type=int,
                        help='input crop size for training')

    # Misc arguments
    parser.add_argument('--seed', default=1337, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--vis', default='./vis',
                        help='folder to output visualization during training')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')
    parser.add_argument('--eval_epoch', type=int, default=1,
                        help='frequency to evaluate')

    # Mode select
    parser.add_argument('--weighted_class', default=True, type=bool, help='set True to use weighted loss')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.batch_size_eval = args.num_gpus * args.batch_size_per_gpu_eval

    # Specify certain arguments
    if args.weighted_class:
        args.enhanced_weight = 2.0
        args.class_weight = np.ones([19], dtype=np.float32)
        enhance_class = [1, 3, 4, 5, 6, 7, 9, 12, 14, 15, 16, 17, 18]
        args.class_weight[enhance_class] = args.enhanced_weight
        args.class_weight = torch.from_numpy(args.class_weight.astype(np.float32))

    args.id += '-ngpus' + str(args.num_gpus)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-imgSize' + str(args.imgSize)
    args.id += '-lr_encoder' + str(args.lr_encoder)
    args.id += '-lr_decoder' + str(args.lr_decoder)
    args.id += '-epoch' + str(args.num_epoch)
    args.id += '-decay' + str(args.weight_decay)
    args.id += '-beta' + str(args.beta)
    if args.weighted_class:
        args.id += '-weighted' + str(args.enhanced_weight) + str(enhance_class)

    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.vis, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
    if not os.path.exists(args.vis):
        os.makedirs(args.vis)

    args.best_err = 2.e10  # initialize with a big number
    args.best_acc = 0
    args.best_mIoU = 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
