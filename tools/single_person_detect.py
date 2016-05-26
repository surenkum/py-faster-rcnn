#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN for Person detection
# --------------------------------------------------------

"""
Demo script showing detections of humans in sample images.
Assumes that only human is present in the image

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import glob
import pdb

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

'''
Pass the net and image that has been read to detect person
'''
def detect_person(net, im,cls_ind=1,cls='person',CONF_THRESH = 0.8):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    NMS_THRESH = 0.3
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    # Filtering by confidence threshold as well
    keep = [ind for ind in keep if cls_scores[ind]>CONF_THRESH]
    if (len(keep)>1):
        sizes = np.zeros((len(keep),))
        for ind,curr_ind in enumerate(keep):
            bbox = dets[curr_ind,:4]
            sizes[ind] = (bbox[3]-bbox[1])*(bbox[2]-bbox[0])
        # Retain only the biggest bounding box
        keep = keep[np.argmax(sizes)]
    
    dets = dets[keep, :]
    return (dets.reshape(1,-1),cls_scores[keep])

def detect_vis_person(net,im,cls_ind=1,cls='person',CONF_THRESH=0.8,im_name='dummy.png'):
    dets,_ = detect_person(net,im,cls_ind=cls_ind,cls=cls,CONF_THRESH=CONF_THRESH)
    vis_detections(im, cls, dets, thresh=CONF_THRESH)
    plt.savefig(im_name[:-4]+'_det'+'.png')

'''
Parse a directory of image with the goal of finding detection threshold that
gives at least one human detection
'''
def find_thresh(imdir_path):
    im_names = parse_im_names(args.imdir_path)
    # Finding the class index of person class in classes being detected
    cls_ind = [ind for ind,name in enumerate(CLASSES) if name=='person'][0]

    max_scores = []
    for im_name in im_names:
        _,curr_score = detect_person(net, cv2.imread(im_name),
            cls_ind = cls_ind)
        max_scores.append(curr_score)
    return min(max_scores)



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('-i',"--input",dest="imdir_path",
			required=True,help='Path to Image directory')

    args = parser.parse_args()

    return args

'''
Read all the images from a folder
'''
def parse_im_names(imdir_path):
    jpg_files = glob.glob(imdir_path+"*.jpg")
    png_files = glob.glob(imdir_path+"*.png")
    jpeg_files = glob.glob(imdir_path+"*.jpeg")
    return jpg_files+png_files+jpeg_files
    
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    # Finding threshold for person detection to be used
    thresh = find_thresh(args.imdir_path)
    print "Will use ",min(thresh,0.8)," as detection threshold for humans"
    # Finding the class index of person class in classes being detected
    cls_ind = [ind for ind,name in enumerate(CLASSES) if name=='person'][0]

    im_names = parse_im_names(args.imdir_path)

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        detect_vis_person(net, cv2.imread(im_name),\
                cls_ind = cls_ind,CONF_THRESH=min(0.8,thresh),im_name=im_name)

    plt.show()
