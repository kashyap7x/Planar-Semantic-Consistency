import os
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.io import loadmat
from scipy.misc import imresize, imsave


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))
    return labelmap_rgb


def accuracy(batch_data, pred):
    (imgs, segs, view2, intrinsics, baseline, disp, infos) = batch_data
    _, preds = torch.max(pred.data.cpu(), dim=1)
    valid = (segs >= 0)
    acc = float(torch.sum(valid * (preds == segs)).item()) / float(torch.sum(valid).item() + 1e-10)
    return acc, float(torch.sum(valid))


def intersectionAndUnion(batch_data, pred, numClass):
    (imgs, segs, view2, intrinsics, baseline, disp, infos) = batch_data
    _, preds = torch.max(pred.data.cpu(), dim=1)

    # compute area intersection
    intersect = preds.clone()
    intersect[torch.ne(preds, segs)] = -1

    area_intersect = torch.histc(intersect.float(),
                                 bins=numClass,
                                 min=0,
                                 max=numClass-1)

    # compute area union:
    preds[torch.lt(segs, 0)] = -1
    area_pred = torch.histc(preds.float(),
                            bins=numClass,
                            min=0,
                            max=numClass-1)
    area_lab = torch.histc(segs.float(),
                           bins=numClass,
                           min=0,
                           max=numClass-1)
    area_union = area_pred + area_lab - area_intersect
    return area_intersect, area_union


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def tensor_grad(tensor, kernel):
    b,c,_,_ = tensor.shape
    kernel = kernel.repeat(1,c,1,1)
    tensor_grad = F.conv2d(tensor, kernel, stride=1, padding=1)
    return tensor_grad


def get_gradients_xy(tensor):
    sobelx = torch.FloatTensor([[0,0,0],[-0.5,0,0.5],[0,0,0]])
    sobely = torch.t(sobelx)
    sobelx_kernel = sobelx.unsqueeze(0).unsqueeze(0).cuda()
    sobely_kernel = sobely.unsqueeze(0).unsqueeze(0).cuda()
    gradx = tensor_grad(tensor, sobelx_kernel)
    grady = tensor_grad(tensor, sobely_kernel)
    return gradx, grady


def planar_smoothness_loss(p_masks, imgs):
    ## Check this implementation
    p_masks_gradx, p_masks_grady = get_gradients_xy(p_masks)

    imgs = F.avg_pool2d(imgs, 8, stride=8)
    imgs_gradx, imgs_grady = get_gradients_xy(imgs)

    weight_x = torch.exp(-torch.mean(torch.abs(imgs_gradx), 1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(imgs_grady), 1, keepdim=True))
    smoothness_x = p_masks_gradx*weight_x 
    smoothness_y = p_masks_grady*weight_y
    smoothness = smoothness_x + smoothness_y
    loss = torch.mean(torch.abs(smoothness))
    return loss


def visualize_masks(batch_data, pred, planar_masks, recons, epoch, args):
    colors = loadmat('colormap.mat')['colors']
    _, c, h, w = planar_masks.shape  
    colors_pl = np.random.randint(0,255,[c,3]).astype(np.uint8)
    
    (imgs, segs, view2, intrs, baseline, disp, infos) = batch_data
    for j in range(len(infos)):
        # get/recover image
        # img = imread(os.path.join(args.root_img, infos[j]))
        img = imgs[j,:3,:,:].clone()
        for t, m, s in zip(img,
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        
        view = view2[j].clone()
        for t, m, s in zip(view,
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        view = (view.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        
        # segmentation
        lab = segs[j].numpy()
        lab_color = colorEncode(lab, colors)

        # planar masks
        pl_mask = np.argmax(planar_masks.data.cpu()[j].numpy(), axis=0)
        pl_color = colorEncode(pl_mask, colors_pl)

        # prediction
        pred_ = np.argmax(pred.data.cpu()[j].numpy(), axis=0)
        pred_color = colorEncode(pred_, colors)
        
        img = imresize(img, (h,w), interp='bilinear')
        lab_color = imresize(lab_color, (h,w), interp='bilinear')
        pred_color = imresize(pred_color, (h,w), interp='bilinear')
        
        recon = recons[j].clone()
        for t, m, s in zip(recon,
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        recon = (recon.cpu().detach().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        
        # aggregate images and save
        im_vis = np.concatenate((img, lab_color, pred_color, pl_color, view, recon),
                                axis=1).astype(np.uint8)
        imsave(os.path.join(args.vis,
                            str(epoch)+"_seg"+infos[j].replace('/', '_')), im_vis)