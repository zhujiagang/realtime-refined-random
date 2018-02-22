""" Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Which was adopated by: Ellis Brown, Max deGroot
    https://github.com/amdegroot/ssd.pytorch

    Further:
    Updated by Gurkirt Singh for ucf101-24 dataset
    Licensed under The MIT License [see LICENSE for details]
"""

import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" #
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, UCF24Detection, AnnotationTransform, detection_collate, CLASSES, BaseTransform, readsplitfile
from utils.augmentations import SSDAugmentation
from layers.modules import RecurrentMultiBoxLoss
from ssd_recurrent_refinement import build_refine_ssd, weights_init
import numpy as np
import time, copy
from utils.evaluation import evaluate_detections
from layers.box_utils import decode, nms, decode2_2center_h_w
from utils import  AverageMeter
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR#, LogLR
from torch.nn.utils import clip_grad_norm
import shutil
from torch.nn import DataParallel
best_prec1 = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

day = (time.strftime('%m-%d', time.localtime(time.time())))
print(day)
relative_path = '/data-sdb/data/jiagang.zhu/'

def main():
    global my_dict, keys, k_len, arr, xxx, args, log_file, best_prec1

    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
    parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
    parser.add_argument('--dataset', default='ucf24', help='pretrained base model')
    parser.add_argument('--ssd_dim', default=300, type=int, help='Input Size for SSD')  # only support 300 now
    parser.add_argument('--modality', default='rgb', type=str,
                        help='INput tyep default rgb options are [rgb,brox,fastOF]')
    parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
    parser.add_argument('--batch_size', default=40, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--max_iter', default=120000, type=int, help='Number of training iterations')
    parser.add_argument('--man_seed', default=123, type=int, help='manualseed for reproduction')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--ngpu', default=1, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--stepvalues', default='70000,90000', type=str,
                        help='iter number when learning rate to be dropped')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.2, type=float, help='Gamma update for SGD')
    parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
    parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
    parser.add_argument('--data_root', default=relative_path + 'realtime/', help='Location of VOC root directory')
    parser.add_argument('--save_root', default=relative_path + 'realtime/saveucf24/',
                        help='Location to save checkpoint models')

    parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
    parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold for evaluation')
    parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
    parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')
    parser.add_argument('--clip_gradient', default=40, type=float, help='gradients clip')
    parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=35, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--eval_freq', default=2, type=int, metavar='N', help='evaluation frequency (default: 5)')
    parser.add_argument('--snapshot_pref', type=str, default="ucf101_vgg16_ssd300_")
    parser.add_argument('--lr_milestones', default=[-2, -5], type=float, help='initial learning rate')
    parser.add_argument('--arch', type=str, default="VGG16")
    parser.add_argument('--Finetune_SSD', default=False, type=str)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])

    print(__file__)
    file_name = (__file__).split('/')[-1]
    file_name = file_name.split('.')[0]
    print(file_name)
    ## Parse arguments
    args = parser.parse_args()
    ## set random seeds
    np.random.seed(args.man_seed)
    torch.manual_seed(args.man_seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.man_seed)

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    args.cfg = v2
    args.train_sets = 'train'
    args.means = (104, 117, 123)
    num_classes = len(CLASSES) + 1
    args.num_classes = num_classes
    args.stepvalues = [int(val) for val in args.stepvalues.split(',')]
    args.loss_reset_step = 30
    args.eval_step = 10000
    args.print_step = 10
    args.data_root += args.dataset + '/'

    ## Define the experiment Name will used to same directory
    day = (time.strftime('%m-%d', time.localtime(time.time())))
    args.snapshot_pref = ('ucf101_CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}').format(args.dataset,
                args.modality, args.batch_size, args.basenet[:-14], int(args.lr*100000)) + '_' + file_name + '_' + day
    print (args.snapshot_pref)

    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)

    net = build_refine_ssd(300, args.num_classes)
    net = torch.nn.DataParallel(net, device_ids=args.gpus)

    if args.Finetune_SSD is True:
        print ("load snapshot")
        pretrained_weights = "/data4/lilin/my_code/realtime/ucf24/rgb-ssd300_ucf24_120000.pth"
        pretrained_dict = torch.load(pretrained_weights)
        model_dict = net.state_dict()  # 1. filter out unnecessary keys
        pretrained_dict_2 = {k: v for k, v in pretrained_dict.items() if k in model_dict } # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict_2) # 3. load the new state dict
    elif args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    elif args.modality == 'fastOF':
        print('Download pretrained brox flow trained model weights and place them at:::=> ',args.data_root + 'ucf24/train_data/brox_wieghts.pth')
        pretrained_weights = args.data_root + 'train_data/brox_wieghts.pth'
        print('Loading base network...')
        net.load_state_dict(torch.load(pretrained_weights))
    else:
        vgg_weights = torch.load(args.data_root +'train_data/' + args.basenet)
        print('Loading base network...')
        net.module.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    # initialize newly added layers' weights with xavier method
    if args.Finetune_SSD is False and args.resume is None:
        print('Initializing weights for extra layers and HEADs...')
        net.module.clstm_1.apply(weights_init)
        net.module.clstm_2.apply(weights_init)
        net.module.extras_r.apply(weights_init)
        net.module.loc_r.apply(weights_init)
        net.module.conf_r.apply(weights_init)

        net.module.extras.apply(weights_init)
        net.module.loc.apply(weights_init)
        net.module.conf.apply(weights_init)

    parameter_dict = dict(net.named_parameters()) # Get parmeter of network in dictionary format wtih name being key
    params = []

    #Set different learning rate to bias layers and set their weight_decay to 0
    for name, param in parameter_dict.items():
        if name.find('vgg') > -1 and int(name.split('.')[2]) < 23:# :and name.find('cell') <= -1
            param.requires_grad = False
            print(name, 'layer parameters will be fixed')
        else:
            if name.find('bias') > -1:
                print(name, 'layer parameters will be trained @ {}'.format(args.lr*2))
                params += [{'params': [param], 'lr': args.lr*2, 'weight_decay': 0}]
            else:
                print(name, 'layer parameters will be trained @ {}'.format(args.lr))
                params += [{'params':[param], 'lr': args.lr, 'weight_decay':args.weight_decay}]

    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = RecurrentMultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    scheduler = None
    # scheduler = LogLR(optimizer, lr_milestones=args.lr_milestones, total_epoch=args.epochs)
    scheduler = MultiStepLR(optimizer, milestones=args.stepvalues, gamma=args.gamma)
    print('Loading Dataset...')
    num_gpu = len(args.gpus)

    rootpath = args.data_root
    imgtype = args.modality
    imagesDir = rootpath + imgtype + '/'
    split = 1
    splitfile = rootpath + 'splitfiles/trainlist{:02d}.txt'.format(split)
    trainvideos = readsplitfile(splitfile)

    splitfile = rootpath + 'splitfiles/testlist{:02d}.txt'.format(split)
    testvideos = readsplitfile(splitfile)

    ####### val dataset does not need shuffle #######
    val_data_loader = []
    len_test = len(testvideos)
    random.shuffle(testvideos)
    for i in range(num_gpu):
        testvideos_temp = testvideos[int(i * len_test / num_gpu):int((i + 1) * len_test / num_gpu)]
        val_dataset = UCF24Detection(args.data_root, 'test', BaseTransform(args.ssd_dim, args.means),
                                     AnnotationTransform(), input_type=args.modality,
                                     full_test=False,
                                     videos=testvideos_temp,
                                     istrain=False)
        val_data_loader.append(data.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, collate_fn=detection_collate, pin_memory=True,
                                               drop_last=True))

    log_file = open(args.save_root + args.snapshot_pref + "_training_" + day + ".log", "w", 1)
    log_file.write(args.snapshot_pref + '\n')

    for arg in vars(args):
        print(arg, getattr(args, arg))
        log_file.write(str(arg) + ': ' + str(getattr(args, arg)) + '\n')

    log_file.write(str(net))

    torch.cuda.synchronize()
    len_train = len(trainvideos)

    for epoch in range(args.start_epoch, args.epochs):
        ####### shuffle train dataset #######
        random.shuffle(trainvideos)
        train_data_loader = []
        for i in range(num_gpu):
            trainvideos_temp = trainvideos[int(i * len_train / num_gpu):int((i + 1) * len_train / num_gpu)]
            train_dataset = UCF24Detection(args.data_root, 'train', SSDAugmentation(args.ssd_dim, args.means),
                                           AnnotationTransform(),
                                           input_type=args.modality,
                                           videos=trainvideos_temp,
                                           istrain=True)
            train_data_loader.append(data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                                     shuffle=False, collate_fn=detection_collate, pin_memory=True, drop_last=True))

        print("Train epoch_size: ", len(train_data_loader))
        print('Train SSD on', train_dataset.name)

        ########## train ###########
        train(train_data_loader, net, criterion, optimizer, scheduler, epoch, num_gpu)

        print('Saving state, epoch:', epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
        }, epoch = epoch)

        #### log lr ###
        # scheduler.step()
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1 or epoch == 0:#
            torch.cuda.synchronize()
            tvs = time.perf_counter()
            mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, epoch, iou_thresh=args.iou_thresh, num_gpu = num_gpu)
            # remember best prec@1 and save checkpoint
            is_best = mAP > best_prec1
            best_prec1 = max(mAP, best_prec1)
            print('Saving state, epoch:', epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
            }, is_best,epoch)

            for ap_str in ap_strs:
                print(ap_str)
                log_file.write(ap_str+'\n')
            ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
            print(ptr_str)
            log_file.write(ptr_str)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
            print(prt_str)
            log_file.write(ptr_str)

    log_file.close()


def train(train_data_loader, net, criterion, optimizer, scheduler, epoch, num_gpu):
    net.train()
    # loss counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    loc_losses = AverageMeter()
    cls_losses = AverageMeter()

    losses_r = AverageMeter()
    loc_losses_r = AverageMeter()
    cls_losses_r = AverageMeter()
    # create batch iterator
    # batch_iterator_new = [[] for i in range(num_gpu)]
    batch_iterator = [[] for i in range(num_gpu)]
    max_x_y = 0
    min_x_y = []
    for i in range(num_gpu):
        batch_iterator[i] = iter(train_data_loader[i])
        min_x_y.append(len(train_data_loader[i]))
        max_x_y = max(max_x_y, len(train_data_loader[i]))
        # print("len: ", len(train_data_loader[i]))

    iter_count = 0
    t0 = time.perf_counter()
    dtype = torch.cuda.FloatTensor
    for iteration in range(max_x_y):
        img_indexs = []
        for ii in range(num_gpu):
            if iteration >= min_x_y[ii]:
                batch_iterator[ii] = iter(train_data_loader[ii])

        images, targets, img_in = next(batch_iterator[0])
        img_indexs.append(img_in)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]

        img = Variable(torch.zeros([1,3,300,300])).type(dtype)
        images = torch.cat((images, img), 0)

        for ii in range(num_gpu - 1):
            img, targ, img_in = next(batch_iterator[ii+1])

            if args.cuda:
                img = Variable(img.cuda())
                targ = [Variable(anno.cuda(), volatile=True) for anno in targ]
            else:
                img = Variable(img)
                targ = [Variable(anno, volatile=True) for anno in targ]

            images = torch.cat((images, img), 0)

            img = Variable(torch.ones([1, 3, 300, 300]) + ii).type(dtype)
            images = torch.cat((images, img), 0)

            for iii in range(len(targ)):
                targets.append(targ[iii])

            img_indexs.append(img_in)

        # forward
        out = net(images, img_indexs)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_l_r, loss_c_r = criterion(out, targets)

        if loss_l is None and loss_l_r is None:
            continue
        elif loss_l is None:
            loss = loss_l_r + loss_c_r
        elif loss_l_r is None:
            loss = loss_l + loss_c
        else:
            loss = loss_l + loss_c + loss_l_r + loss_c_r

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(net.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()
        if scheduler is not None:
             scheduler.step()

        if loss_l is not None:
            loc_loss = loss_l.data[0]
            conf_loss = loss_c.data[0]
        if loss_l_r is not None:
            loc_loss_r = loss_l_r.data[0]
            conf_loss_r = loss_c_r.data[0]

        loc_losses.update(loc_loss)
        cls_losses.update(conf_loss)

        loc_losses_r.update(loc_loss_r)
        cls_losses_r.update(conf_loss_r)

        losses.update((loc_loss + conf_loss) / 2.0)
        losses_r.update((loc_loss_r + conf_loss_r) / 2.0)

        if iteration % args.print_step == 0 and iteration > 0:

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_time.update(t1 - t0)

            print_line = 'Epoch {:02d}/{:02d} Iteration {:06d}/{:06d} loc-loss {:.3f}({:.3f}) cls-loss {:.3f}({:.3f}) ' \
                         'average-loss {:.3f}({:.3f}) Timer {:0.3f}({:0.3f}) lr {:0.5f}'.format(
                epoch, args.epochs, iteration, max_x_y, loc_losses.val, loc_losses.avg, cls_losses.val,
                cls_losses.avg, losses.val, losses.avg, batch_time.val, batch_time.avg, args.lr)

            print_line_r = 'Epoch {:02d}/{:02d} Iteration {:06d}/{:06d} loc-loss-refined {:.3f}({:.3f}) cls-loss-refined {:.3f}({:.3f}) ' \
                         'average-loss-refined {:.3f}({:.3f}) Timer {:0.3f}({:0.3f}) lr {:0.5f}'.format(
                epoch, args.epochs, iteration, len(train_data_loader), loc_losses_r.val, loc_losses_r.avg, cls_losses_r.val,
                cls_losses_r.avg, losses_r.val, losses_r.avg, batch_time.val, batch_time.avg, args.lr)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            log_file.write(print_line + '\n')
            print(print_line)
            log_file.write(print_line_r + '\n')
            print(print_line_r)

            iter_count += 1
            if iter_count % args.loss_reset_step == 0 and iter_count > 0:
                loc_losses.reset()
                cls_losses.reset()
                losses.reset()
                loc_losses_r.reset()
                cls_losses_r.reset()
                losses_r.reset()
                batch_time.reset()
                print('Reset accumulators of ', args.snapshot_pref, ' at', iter_count * args.print_step)
                iter_count = 0


def validate(args, net, val_data_loader, val_dataset, epoch, iou_thresh=0.5, num_gpu = 1):
    """Test a SSD network on an image database."""
    print('Validating at ', epoch)
    num_images = len(val_dataset)
    num_classes = args.num_classes

    det_boxes = [[] for _ in range(len(CLASSES))]
    gt_boxes = []
    print_time = True
    val_step = 100
    count = 0
    net.eval()  # switch net to evaluation modelen(val_data_loader)-2,
    torch.cuda.synchronize()
    ts = time.perf_counter()

    # create batch iterator

    batch_iterator = [[] for i in range(num_gpu)]
    max_x_y = 0
    min_x_y = []
    for i in range(num_gpu):
        batch_iterator[i] = iter(val_data_loader[i])
        min_x_y.append(len(val_data_loader[i]))
        max_x_y = max(max_x_y, len(val_data_loader[i]))
        # print("len: ", len(train_data_loader[i]))

    iter_count = 0
    t0 = time.perf_counter()
    dtype = torch.cuda.FloatTensor
    for val_itr in range(max_x_y):
        img_indexs = []
        for ii in range(num_gpu):
            if val_itr >= min_x_y[ii]:
                batch_iterator[ii] = iter(val_data_loader[ii])

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        img_indexs = []
        images, targets, img_in = next(batch_iterator[0])
        img_indexs.append(img_in)

        img = torch.zeros([1, 3, 300, 300])
        images = torch.cat((images, img.type_as(images)), 0)

        for ii in range(num_gpu-1):
            img, targ, img_in = next(batch_iterator[ii+1])
            images = torch.cat((images, img), 0)
            img = (torch.ones([1, 3, 300, 300])  + ii)
            images = torch.cat((images, img.type_as(images)), 0)
            for iii in range(len(targ)):
                targets.append(targ[iii])

            img_indexs.append(img_in)

        batch_size = images.size(0) - num_gpu
        height, width = images.size(2), images.size(3)

        if args.cuda:
            images = Variable(images.cuda(), volatile=True)

        output = net(images, img_indexs)

        loc_data = output[0]
        conf_preds = output[1]
        loc_data_r = output[2]
        conf_preds_r = output[3]
        prior_data = output[4]
        prior_data = prior_data[:loc_data.size(1), :]

        if print_time and val_itr%val_step == 0:
            torch.cuda.synchronize()
            tf = time.perf_counter()
            print('Forward Time {:0.3f}'.format(tf-t1))
        for b in range(batch_size):
            gt = targets[b].numpy()
            gt[:,0] *= width
            gt[:,2] *= width
            gt[:,1] *= height
            gt[:,3] *= height
            gt_boxes.append(gt)

            defaults_r = loc_data[b].data
            decode_defaults_r = decode2_2center_h_w(defaults_r, prior_data.data, args.cfg['variance'])

            decoded_boxes = decode(loc_data_r[b].data, decode_defaults_r, args.cfg['variance']).clone()
            conf_scores = net.module.softmax(conf_preds_r[b]).data.clone()

            for cl_ind in range(1, num_classes):
                scores = conf_scores[:, cl_ind].squeeze()
                c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
                scores = scores[c_mask].squeeze()
                # print('scores size',scores.size())
                if scores.dim() == 0:
                    # print(len(''), ' dim ==0 ')
                    det_boxes[cl_ind - 1].append(np.asarray([]))
                    continue
                boxes = decoded_boxes.clone()
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes = boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
                scores = scores[ids[:counts]].cpu().numpy()
                boxes = boxes[ids[:counts]].cpu().numpy()
                # print('boxes sahpe',boxes.shape)
                boxes[:,0] *= width
                boxes[:,2] *= width
                boxes[:,1] *= height
                boxes[:,3] *= height

                for ik in range(boxes.shape[0]):
                    boxes[ik, 0] = max(0, boxes[ik, 0])
                    boxes[ik, 2] = min(width, boxes[ik, 2])
                    boxes[ik, 1] = max(0, boxes[ik, 1])
                    boxes[ik, 3] = min(height, boxes[ik, 3])

                cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)

                det_boxes[cl_ind-1].append(cls_dets)
            count += 1
        if val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
            torch.cuda.synchronize()
            ts = time.perf_counter()
        if print_time and val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('NMS stuff Time {:0.3f}'.format(te - tf))
    print('Evaluating detections for epoch number ', epoch)
    return evaluate_detections(gt_boxes, det_boxes, CLASSES, iou_thresh=iou_thresh)

def save_checkpoint(state, is_best = False, epoch = 0):
    file_name = (__file__).split('/')[-1]
    file_name = file_name.split('.')[0]
    print(file_name)
    day = (time.strftime('%m-%d', time.localtime(time.time())))
    args.snapshot_pref = ('ucf101_CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}').format(args.dataset,
                args.modality, args.batch_size, args.basenet[:-14], int(args.lr*100000)) + '_' + file_name + '_' + day
    print (args.snapshot_pref)
    snapshot = args.save_root + args.snapshot_pref + '_epoch_' + str(epoch)
    filename = snapshot + '_checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        best_name = snapshot + '_model_best.pth.tar'
        shutil.copyfile(filename, best_name)

if __name__ == '__main__':
    main()