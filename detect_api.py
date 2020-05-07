import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from .opt import opt

from .dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from .yolo.util import write_results, dynamic_write_results
from .SPPE.src.main_fast_inference import *

import os
import sys
from tqdm import tqdm
import time
from .fn import getTime

from .pPose_nms import pose_nms, write_json

args = opt
args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

def detect(img_files,
           yolo_cfg="./yolo/cfg/yolov3-spp.cfg",
           yolo_path="./models/yolo/yolov3-spp.weights",
           sppe_path="./models/sppe/duc_se.pth",
           batch_size=1,
           pose_batch=80,
           profile=False,
           fast_inference=True):

    # Load input images
    data_loader = ImageLoader(img_files, batchSize=batch_size, format='yolo').start()

    # Load detection loader
    # print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=batch_size,
                                 yolo_cfg=yolo_cfg, yolo_path=yolo_path).start()
    det_processor = DetectionProcessor(det_loader).start()

    # Load pose model
    pose_dataset = Mscoco()
    if fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset, sppe_path=sppe_path)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset, sppe_path=sppe_path)
    pose_model.cuda()
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    writer = DataWriter(args.save_video).start()

    data_len = data_loader.length()
    if profile:
        im_names_desc = tqdm(range(data_len))
    else:
        im_names_desc = range(data_len)

    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation

            datalen = inps.size(0)
            leftover = 0
            if (datalen) % pose_batch:
                leftover = 1
            num_batches = datalen // pose_batch + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j*pose_batch:min((j +  1)*pose_batch, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)
            hm = hm.cpu()
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        if profile:
            # TQDM
            im_names_desc.set_description(
            'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            )

    # print('===========================> Finish Model Running.')
    while(writer.running()):
        pass
    writer.stop()
    final_result = writer.results()
    return final_result
