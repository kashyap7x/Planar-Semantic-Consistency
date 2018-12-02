import torch
import torch.nn as nn
import math
from lib.nn import SynchronizedBatchNorm2d


class NovelViewHomography(nn.Module): # Actually warp in train.h
    def __init__(self):
        super(NovelViewHomography, self).__init__()

    def forward(self, img, planar_masks, disp, intrs, baseline):
        """
        Args: img(bxcxhxw), planar_masks(bx1xhxw), depth(bx1xhxw), normal()
        """
        R, t = self.get_Rt(baseline)

        # Get depth and normal from disp 
        depth = self.disp2depth(disp, intrs, baseline)
        normal = self.depth2normal(depth)
        planar_depth, planar_normal = get_planar_depth_normal(planar_masks, depth, normal)
        H = self.get_H(planar_depth, planar_normal, R, t, intrs)
        tx_img =  self.warpH(img, H)
        return tx_img
        # for i_m in range(args.num_planes):
        #     grid_size = img.size()
        #     grid = F.affine_grid(H, grid_size)
        #     tx_img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros')
        # return tx_img

    def get_planar_depth_normal(self, planar_masks, depth, normal):
        b, c, h, w  = planar_masks.shape
        flat_masks = planar_masks.view(b, c, w*h)
        planar_weights = F.softmax(flat_masks, -1)
        flat_depth = depth.view(b, w*h)
        flat_normal = normal.view(b, 3, w*h)
        planar_depth = torch.sum(planar_weights*flat_depth,-1)
        planar_normal = torch.sum(planar_weights*flat_normal,-1)
        return planar_depth, planar_normal

    
    def get_Rt(self, baseline):
        batch_size = size(baseline,0)
        R = torch.eye(3)
        R = R.repeat(batch_size,1)
        R = Variable(R).cuda()
        t = torch.zeros(batch_size,3,1).cuda()
        t[2] = -baseline
        t = Variable(t).cuda()
        return R, t

    def getH(self, planar_depth, planar_normal, R, t, K):
        """
        Extracts H using m depths, normals, planes and pose
        # Args: planar_depth(bxm), planar_normal(bxmx3x1), R(bx3x3), t(bx3x1), K(bx3x3)
        
        Output:	H(bxmx3x3)
        """

        n = planar_normal
        d = planar_depth
        n_t = torch.transpose(n,-2,-1)
        K_inv = b_inv(K)
        H_cal = R-torch.bmm(t,n_t/d)
        H = torch.bmm(K,torch.bmm(H_cal,K_inv))
        return H



    def b_inv(self, b_mat):
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv, _ = torch.gesv(eye, b_mat)
        return b_inv


    def getPlaneEq(self, normal, depth, masks):
        """
        VERIFY EACH STEP
        Args: normal(mx3xhxw), depth(mxhxw), masks(mxhxw)
        Output: plane_eq(mx4)
        """
        masked_normal = normal*masks # verify broadcasting
        masked_depth = depth*masks
        n = torch.sum(masked_normal.view(args.num_planes, 3, -1), -1)/torch.sum(masks.view(args.num_planes, -1), -1)
        d = torch.sum(masked_depth.view(args.num_planes, -1),-1)/torch.sum(masks.view(args.num_planes, -1), -1)
        return n, d


    def warpH(self, img, H):
        """
        Args: img(bxcxhxw), H(bxmx3x3)
        """
        tx_img = torch.zeros(img.shape)
        for i_m in range(args.num_planes):
        	H_plane = H[:,i_m,:2,3].squeeze()
            grid_size = img.size()
            grid = F.affine_grid(H_plane, grid_size)
            tx_img(:,i_m,:,:) = F.grid_sample(img(:,i_m,:,:), grid, mode='bilinear', padding_mode='zeros')
        return tx_img

    def disp2depth(self, disp, intrs, baseline):
        fx = intrs[:,0, 0]
        fy = intrs[:,1, 1]
        focal = (fx+fy)/2
        depth = baseline * focal/(disparity*disp.size(2));
        print('Width of image for disparity scaling: %d' %disp.size(2))
        return depth

    def tensor_grad(self, depth, kernel):
        depth_grad = F.conv2d(depth, kernel, padding=1)
        return depth_grad

    def depth2normals(self, depth):
        # depth(B,H,W)
        sobelx = np.array[0,0,0;
                          -1,0,1;
                          0,0,0]
        sobely = sobelx.T 
        sobelx_kernel = torch.FloatTensor(sobelx).unsqueeze(0).cuda()
        sobely_kernel = torch.FloatTensor(sobely).unsqueeze(0).cuda()
        depth_gradx = self.tensor_grad(depth, sobelx_kernel)
        depth_grady = self.tensor_grad(depth, sobely_kernel)
        dir_map = torch.stack([-depth_gradx, -depth_grady, torch.ones(size(depth_grad_x))],1)
        norm_factor = torch.sqrt(torch.sum(dir_map**2,1))
        normal_map = dir_map/norm_factor
        return normal_map


class ModelBuilder():
    def build_encoder(self, arch='resnet', weights=''):
        if arch == 'resnet':
            net_encoder = ResnetEncoder(resnet18())
        if len(weights) > 0:
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage))
        return net_encoder

    def build_decoder(self, arch='ppm', num_class=19, num_plane=100, use_softmax=True, weights=''):
        if arch == 'c1':
            net_decoder = C1Decoder(num_class, num_plane, use_softmax)
        elif arch == 'ppm':
            net_decoder = PPMDecoder(num_class, use_softmax)
        if len(weights) > 0:
            pretrained_dict = torch.load(weights, map_location=lambda storage, loc: storage)
            net_decoder.load_state_dict(pretrained_dict, strict=False)
        return net_decoder


class ResnetEncoder(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, dropout2d=False):
        super(ResnetEncoder, self).__init__()
        self.dropout2d = dropout2d
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        if self.dropout2d:
            self.dropout = nn.Dropout2d(0.5)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        y = self.maxpool(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.dropout2d:
            x = self.dropout(x)
        return x, y


class PPMDecoder(nn.Module):
    def __init__(self, num_class=19, use_softmax=True, pool_scales=(1, 2, 3, 6)):
        super(PPMDecoder, self).__init__()
        self.use_softmax = use_softmax

        self.psp = []
        for scale in pool_scales:
            self.psp.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(512, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.psp = nn.ModuleList(self.psp)

        self.conv_final = nn.Sequential(
            nn.Conv2d(512+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()
        psp_out = [x]
        for pool_scale in self.psp:
            psp_out.append(nn.functional.upsample(
                pool_scale(x),
                (input_size[2], input_size[3]),
                mode='bilinear'))
        psp_out = torch.cat(psp_out, 1)

        x = self.conv_final(psp_out)
        x = nn.functional.upsample(x, size=(input_size[2]*8, input_size[3]*8), mode='bilinear')

        if self.use_softmax:
            x = nn.functional.log_softmax(x, dim=1)

        return x


class C1Decoder(nn.Module):
    def __init__(self, num_class = 19, num_plane = 100, use_softmax=False):
        super(C1Decoder, self).__init__()
        self.use_softmax = use_softmax
        
        self.avgpool = nn.AvgPool2d(8, stride=8)
        
        # last conv
        self.conv_last = nn.Conv2d(64 + num_class, num_plane, 1, 1, 0)

    def forward(self, x, probs):
        input_size = x.size()
        probs = self.avgpool(probs)
        x = self.conv_last(torch.cat([x,probs],1))
        
        x = nn.functional.upsample(x, size=(input_size[2]*8, input_size[3]*8), mode='bilinear')
        if self.use_softmax:
            x = nn.functional.log_softmax(x, dim=1)

        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = SynchronizedBatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = SynchronizedBatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model