
""" SSD network Classes

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

Updated by Gurkirt Singh for ucf101-24 dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v2
import os
from convlstm import CLSTM
from congru import CGRU

class RefineSSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, base, extras, extras_refinement, head, head_r, num_classes):#, use_gru = False
        super(RefineSSD, self).__init__()

        self.num_classes = num_classes
        self.binary_classes = 2
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.num_priors = self.priors.size(0)
        self.size = 300
        self.hidden_states = [[[], [], [], [], [], []],
                              [[], [], [], [], [], []],
                              [[], [], [], [], [], []],
                              [[], [], [], [], [], []]]

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.clstm_1 = nn.ModuleList([CLSTM(512, 512, 3, stride=1, padding=1)])
        self.clstm_2 = nn.ModuleList([CLSTM(1024, 1024, 3, stride=1, padding=1)])

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.L2Norm_r = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.extras_r = nn.ModuleList(extras_refinement)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.loc_r = nn.ModuleList(head_r[0])
        self.conf_r = nn.ModuleList(head_r[1])

        self.softmax = nn.Softmax().cuda()
        # self.detect = Detect(num_classes, 0, 200, 0.001, 0.45)

    def forward(self, x, indexes):

        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        last_img = x[-1][0][0][0].data
        last = int(last_img.cpu().numpy()[0])
        x = x[:-1]
        indexes = indexes[last]
        reset_indexes = []
        reset_cnt = 0
        for item in indexes:
            if item[1] is True:
                reset_indexes.append(reset_cnt)
            reset_cnt += 1

        reset_group = []
        if len(reset_indexes) > 0:
            reset_indexes.append(len(indexes))
            for i in range(len(reset_indexes)-1):
                reset_group.append([reset_indexes[i], reset_indexes[i+1]])

        sources = list()
        loc = list()
        conf = list()

        sources_r = list()
        loc_r = list()
        conf_r = list()

        # apply vgg up to conv4_3 relu
        x_21 = []
        for k in range(23):
            x = self.vgg[k](x)
            if k == 21:
                x_21 = x

        def convlstm_forward(x, CLSTM, hidden_states, hi):
            len_r = len(reset_group)
            if len_r > 0:
                yyy = []
                if reset_group[0][0] > 0:
                    temp_x = x[0:reset_group[0][0]]
                    xx = temp_x.view(-1, temp_x.size(0), temp_x.size(1), temp_x.size(2), temp_x.size(3))
                    len_hid = len(hidden_states[hi])
                    if len_hid > 0:
                        temp = [(Variable(hidden_states[hi][0]).cuda(),
                                 Variable(hidden_states[hi][1]).cuda())]
                        xxx = CLSTM(xx, temp)
                    else:
                        xxx = CLSTM(xx)
                    hidden_states[hi] = [xxx[0][0][0].data, xxx[0][0][1].data]
                    yyy.append(xxx[1].data)

                for item in reset_group:
                    hidden_states[hi] = []
                    temp_x = x[item[0]:item[1]]
                    xx = temp_x.view(-1, temp_x.size(0), temp_x.size(1), temp_x.size(2), temp_x.size(3))
                    xxx = CLSTM(xx)
                    hidden_states[hi] = [xxx[0][0][0].data, xxx[0][0][1].data]
                    yyy.append(xxx[1].data)

                third_tensor = yyy[0]
                if len(yyy) > 1:
                    for yitem in yyy[1:]:
                        third_tensor = torch.cat((third_tensor, yitem), 0)

                yyyy = Variable(third_tensor).cuda()
                yyyy = F.relu(yyyy, inplace=True)
                x = yyyy.view(-1, *yyyy.size()[2:])
                return x
            else:
                xx = x.view(-1, x.size(0), x.size(1), x.size(2), x.size(3))
                len_hid = len(hidden_states[hi])
                if len_hid > 0:
                    temp = [(Variable(hidden_states[hi][0]).cuda(), Variable(hidden_states[hi][1]).cuda())]
                    xxx = CLSTM(xx, temp)
                else:
                    xxx = CLSTM(xx)

                yyy = xxx[1]
                hidden_states[hi] = [xxx[0][0][0].data, xxx[0][0][1].data]

                yyyy = F.relu(yyy, inplace=True)
                x = yyyy.view(-1, *yyyy.size()[2:])
                return x

        def convgru_forward(x, CGRU, hidden_states, hi):
            len_r = len(reset_group)
            if len_r > 0:
                yyy = []
                if reset_group[0][0] > 0:
                    temp_x = x[0:reset_group[0][0]]
                    xx = temp_x.view(-1, temp_x.size(0), temp_x.size(1), temp_x.size(2), temp_x.size(3))
                    len_hid = len(hidden_states[hi])
                    if len_hid > 0:
                        temp = [(Variable(hidden_states[hi]).cuda())]
                        xxx = CGRU(xx, temp)
                    else:
                        xxx = CGRU(xx)
                    hidden_states[hi] = xxx[0][0].data
                    yyy.append(xxx[1].data)

                for item in reset_group:
                    hidden_states[hi] = []
                    temp_x = x[item[0]:item[1]]
                    xx = temp_x.view(-1, temp_x.size(0), temp_x.size(1), temp_x.size(2), temp_x.size(3))
                    xxx = CGRU(xx)
                    hidden_states[hi] = xxx[0][0].data
                    yyy.append(xxx[1].data)

                third_tensor = yyy[0]
                if len(yyy) > 1:
                    for yitem in yyy[1:]:
                        third_tensor = torch.cat((third_tensor, yitem), 0)

                yyyy = Variable(third_tensor).cuda()
                yyyy = F.relu(yyyy, inplace=True)
                x = yyyy.view(x.size())
                return x
            else:
                xx = x.view(-1, x.size(0), x.size(1), x.size(2), x.size(3))
                len_hid = len(hidden_states[hi])
                if len_hid > 0:
                    temp = [(Variable(hidden_states[hi]).cuda())]
                    xxx = CGRU(xx, temp)
                else:
                    xxx = CGRU(xx)

                yyy = xxx[1]
                hidden_states[hi] = xxx[0][0].data

                yyyy = F.relu(yyy, inplace=True)
                x = yyyy.view(x.size())
                return x

        hi = 0
        ##  apply convlstm on conv4_3

        s = self.L2Norm(x)
        sources.append(s)

        x_22_r = convlstm_forward(x_21, self.clstm_1[0], self.hidden_states[last], hi)
        s_r_1 = self.L2Norm_r(x_22_r)
        sources_r.append(s_r_1)

        hi += 1
        x_vgg_last = []
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
            if k == (len(self.vgg) - 2):
                x_vgg_last = x

        sources.append(x)

        s_r_2 = convlstm_forward(x_vgg_last, self.clstm_2[0], self.hidden_states[last], hi)
        sources_r.append(s_r_2)

        hi += 1
        x_extras = x

        # apply extra layers and cache source layer outputs
        cnt = 0
        for k, v in enumerate(self.extras):
            x_extras_main = F.relu(v(x_extras), inplace=True)
            if k % 2 == 1:
                sources.append(x_extras_main)
                # apply refinement extra layers and cache source layer outputs
                x_extras_r = convlstm_forward(x_extras, self.extras_r[cnt], self.hidden_states[last], hi)
                sources_r.append(x_extras_r)
                cnt += 1
                hi += 1
            x_extras = x_extras_main

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        for (x, l, c) in zip(sources_r, self.loc_r, self.conf_r):
            loc_r.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_r.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc_r = torch.cat([o.view(o.size(0), -1) for o in loc_r], 1)
        conf_r = torch.cat([o.view(o.size(0), -1) for o in conf_r], 1)

        output = (loc.view(loc.size(0), -1, 4),
                  conf.view(conf.size(0), -1, self.binary_classes),
                  loc_r.view(loc_r.size(0), -1, 4),
                  conf_r.view(conf_r.size(0), -1, self.num_classes),
                  self.priors)
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    cnt = 0
    # clstm_1 = CLSTM(512, 512, 3, 1)
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            cnt += 1
            in_channels = v

        cnt += 1

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    print (cfg[0])
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            in_c = 0
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
                in_c = cfg[k + 1]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                in_c = v

            flag = not flag
        in_channels = v
    return layers

def add_extras_recurrent():
    # recurrent Extra layers added to VGG for feature scaling
    layers = []

    # [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    # layers += [nn.Conv2d(1024, 256, kernel_size=(1, 1))]
    layers += [CLSTM(256, 512, filter_size=3, stride=2, padding=1)]
    # layers += [nn.Conv2d(512, 128, kernel_size=(1, 1))]
    layers += [CLSTM(128, 256, filter_size=3, stride=2, padding=1)]
    # layers += [nn.Conv2d(256, 128, kernel_size=(1, 1))]
    layers += [CLSTM(128, 256, filter_size=3, stride=1)]
    # layers += [nn.Conv2d(256, 128, kernel_size=(1, 1))]
    layers += [CLSTM(128, 256, filter_size=3, stride=1)]

    return layers


def recurrent_multibox(vgg, extra_layers, recurrent_extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers_r = []
    conf_layers_r = []
    vgg_source = [24, -2]
    binary_classes = 2
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]

        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * binary_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * binary_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(vgg_source):
        loc_layers_r += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]

        conf_layers_r += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(recurrent_extra_layers[0::1], 2):
        loc_layers_r += [nn.Conv2d(v.num_features, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers_r += [nn.Conv2d(v.num_features, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]

    return vgg, extra_layers, recurrent_extra_layers, (loc_layers, conf_layers), (loc_layers_r, conf_layers_r)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}

extras_r = {
    '300': [512, 256, 256, 256],
    '512': [],
}

mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}

mbox_4 = {
    '300': [6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_refine_ssd(size=300, num_classes=21, use_gru = False):

    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return

    return RefineSSD(*recurrent_multibox(
                                vgg(base[str(size)], 3),
                                add_extras(extras[str(size)], 1024),
                                add_extras_recurrent(),
                                mbox[str(size)],
                                num_classes),
                    num_classes)

import torch.nn.init as init

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def weights_init_for_extra(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)