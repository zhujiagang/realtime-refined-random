import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v2 as cfg
from ..box_utils import match, match_binary,log_sum_exp, decode, decode2_2center_h_w

class RecurrentMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(RecurrentMultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.binary_classes = 2
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, loc_data_r, conf_data_r, priors = predictions
        priors_loc_data = loc_data
        # priors_loc_data = loc_data.view(-1, 4)

        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        # match spatial refined anchors and ground truth boxes
        loc_t_r = torch.Tensor(num, num_priors, 4)
        conf_t_r = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            defaults_r = priors_loc_data[idx].data

            match_binary(self.threshold, truths, defaults, self.variance, labels,
                         loc_t, conf_t, idx)

            decode_defaults_r = decode2_2center_h_w(defaults_r, defaults, self.variance)

            decode_defaults_r.clamp_(max=1, min=0)

            match(self.threshold, truths, decode_defaults_r, self.variance, labels,
                  loc_t_r, conf_t_r, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            loc_t_r = loc_t_r.cuda()
            conf_t_r = conf_t_r.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        loc_t_r = Variable(loc_t_r, requires_grad=False)
        conf_t_r = Variable(conf_t_r, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(keepdim=True)

        pos_r = conf_t_r > 0
        num_pos_r = pos_r.sum(keepdim=True)


        softmax = nn.Softmax2d().cuda()
        xxxx = conf_data.view(1, conf_data.size(0), conf_data.size(1), conf_data.size(2))
        y = softmax(xxxx.permute(0, 3, 1, 2))
        y = y.permute(0, 2, 3, 1)
        y = torch.squeeze(y, 0)
        refined_anchor_neg = y[:,:,0]

        refined_anchor = refined_anchor_neg > 0.99
        num_refined_anchor = refined_anchor.sum(keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        ### single frame anchor refinement module
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining on binary classification
        batch_conf = conf_data.view(-1, self.binary_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.binary_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        ### temporal action detection module
        pos_idx_r = pos_r.unsqueeze(pos_r.dim()).expand_as(loc_data_r)
        loc_p_r = loc_data_r[pos_idx_r].view(-1, 4)
        loc_t_r = loc_t_r[pos_idx_r].view(-1, 4)
        loss_l_r = F.smooth_l1_loss(loc_p_r, loc_t_r, size_average=False)

        # Compute max conf across batch for hard negative mining for multi-class classification
        batch_conf_r = conf_data_r.view(-1, self.num_classes)
        loss_c_r = log_sum_exp(batch_conf_r) - batch_conf_r.gather(1, conf_t_r.view(-1, 1))

        loss_c_r[pos_r] = 0  # filter out pos boxes for now
        loss_c_r[refined_anchor] = 0  # filter out neg boxes with high confidence

        loss_c_r = loss_c_r.view(num, -1)
        _, loss_idx_r = loss_c_r.sort(1, descending=True)
        _, idx_rank_r = loss_idx_r.sort(1)
        num_pos_r = pos_r.long().sum(1, keepdim=True)
        num_neg_r = torch.clamp(self.negpos_ratio*num_pos_r, max=pos_r.size(1)-1)
        neg_r = idx_rank_r < num_neg_r.expand_as(idx_rank_r)

        pos_idx_r = pos_r.unsqueeze(2).expand_as(conf_data_r)
        neg_idx_r = neg_r.unsqueeze(2).expand_as(conf_data_r)
        conf_p_r = conf_data_r[(pos_idx_r+neg_idx_r).gt(0)].view(-1, self.num_classes)
        targets_weighted_r = conf_t_r[(pos_r+neg_r).gt(0)]
        loss_c_r = F.cross_entropy(conf_p_r, targets_weighted_r, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        if N == 0:
            loss_l = None #loss_l_last
            loss_c = None #loss_c_last
            print ("zero positive anchor at single frame refinement module")
        else:
            loss_l /= N
            loss_c /= N

        N_r = num_pos_r.data.sum()
        if N_r == 0:
            loss_l_r = None #loss_l_r_last
            loss_c_r = None #loss_c_r_last
            print("zero positive anchor at temporal action detection module")
        else:
            loss_l_r /= N_r
            loss_c_r /= N_r

        # print (N, N_r)

        return loss_l, loss_c, loss_l_r, loss_c_r