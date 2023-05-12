# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os.path as osp

import cv2
import time
import numpy as np
from PIL import Image
import scipy.io as sio

from data.choose_config import cfg
cfg = cfg.cfg

from torch.autograd import Variable
from utils.augmentations import to_chw_bgr

from importlib import import_module

import warnings
warnings.filterwarnings("ignore")

HOME = os.environ['DATA_HOME']

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='s3fd evaluatuon wider')
parser.add_argument('--model', type=str,
                    default='./weights/rpool_face_c.pth', help='trained model')
parser.add_argument('--thresh', default=0.05, type=float,
                    help='Final confidence threshold')
parser.add_argument('--multigpu',
                    default=False, type=str2bool,
                    help='Specify whether model was trained with multigpu')
parser.add_argument('--model_arch',
                    default='RPool_Face_C', type=str,
                    choices=['RPool_Face_C', 'RPool_Face_Quant', 'RPool_Face_QVGA_monochrome'],
                    help='choose architecture among rpool variants')
parser.add_argument('--save_folder', type=str,
                    default='rpool_face_predictions', help='folder for saving predictions')
parser.add_argument('--subset', type=str,
                    default='val',
                    choices=['val', 'test'],
                    help='choose which set to run testing on')

args = parser.parse_args()


if use_cuda := torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect_face(net, img, shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink,
                         interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(img)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    if cfg.IS_MONOCHROME == True:
        x = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    else:
        x = torch.from_numpy(x).unsqueeze(0)

    if use_cuda:
        x = x.cuda()

    y,_,_ = net(x)
    detections = y.data
    detections = detections.cpu().numpy()

    det_conf = detections[0, 1, :, 0]
    det_xmin = img.shape[1] * detections[0, 1, :, 1] / shrink
    det_ymin = img.shape[0] * detections[0, 1, :, 2] / shrink
    det_xmax = img.shape[1] * detections[0, 1, :, 3] / shrink
    det_ymax = img.shape[0] * detections[0, 1, :, 4] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= args.thresh)[0]
    det = det[keep_index, :]

    return det


def multi_scale_test(net, image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(net, image, st)
    index = np.where(np.maximum(
        det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (
        st + max_im_shrink) / 2
    det_b = detect_face(net, image, bt)

    # enlarge small image x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(net, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(net, image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
    else:
        index = np.where(np.maximum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
    det_b = det_b[index, :]
    return det_s, det_b


def flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(net, image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets


def get_data():
    subset = args.subset

    WIDER_ROOT = os.path.join(HOME, 'WIDER_FACE')
    if subset == 'val':
        wider_face = sio.loadmat(
            os.path.join(WIDER_ROOT, 'wider_face_split',
                               'wider_face_val.mat'))
    else:
        wider_face = sio.loadmat(
            os.path.join(WIDER_ROOT, 'wider_face_split',
                               'wider_face_test.mat'))
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face

    imgs_path = os.path.join(WIDER_ROOT, f'WIDER_{subset}', 'images')
    save_path = f'./{args.save_folder}'

    return event_list, file_list, imgs_path, save_path

if __name__ == '__main__':
    event_list, file_list, imgs_path, save_path = get_data()
    cfg.USE_NMS = False

    module = import_module(f'models.{args.model_arch}')
    net = module.build_s3fd('test', cfg.NUM_CLASSES)

    if args.multigpu == True:
        net = torch.nn.DataParallel(net)


    checkpoint_dict = torch.load(args.model)

    model_dict = net.state_dict()


    model_dict.update(checkpoint_dict)
    net.load_state_dict(model_dict)


    net.eval()


    if use_cuda:
        net.cuda()
        cudnn.benckmark = True


    counter = 0

    for index, event in enumerate(event_list):
        filelist = file_list[index][0]
        path = os.path.join(save_path, str(event[0][0]))#.encode('utf-8'))
        if not os.path.exists(path):
            os.makedirs(path)

        for file in filelist:
            im_name = str(file[0][0])#.encode('utf-8')
            in_file = os.path.join(imgs_path, event[0][0], f'{im_name[:]}.jpg')
            img = Image.open(in_file)
            if img.mode == 'L':
                img = img.convert('RGB')
            img = np.array(img)


            max_im_shrink = np.sqrt(
                1700 * 1200 / (img.shape[0] * img.shape[1]))

            shrink = min(max_im_shrink, 1)
            counter += 1

            t1 = time.time()
            det0 = detect_face(net, img, shrink)

            det1 = flip_test(net, img, shrink)    # flip test
            [det2, det3] = multi_scale_test(net, img, max_im_shrink)

            det = np.row_stack((det0, det1, det2, det3))
            dets = bbox_vote(det)

            t2 = time.time()
            print('Detect %04d th image costs %.4f' % (counter, t2 - t1))

            fout = open(
                osp.join(save_path, str(event[0][0]), f'{im_name}.txt'), 'w'
            )
            fout.write('{:s}\n'.format(f'{str(event[0][0])}/{im_name}.jpg'))
            fout.write('{:d}\n'.format(dets.shape[0]))
            for i in range(dets.shape[0]):
                xmin = dets[i][0]
                ymin = dets[i][1]
                xmax = dets[i][2]
                ymax = dets[i][3]
                score = dets[i][4]
                fout.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                           format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))
