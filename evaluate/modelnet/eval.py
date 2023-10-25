import os
import random
import numpy as np
import logging
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from utils import config2
from tqdm import tqdm
from datasets import ModelNetDataset

random.seed(123)
np.random.seed(123)

def get_parser():
    parser = argparse.ArgumentParser(description='ModelNet Dataset Instances')
    parser.add_argument('--config', type=str, default='configs\\modelnet\\modelnet.yaml', help='config file')
    parser.add_argument('opts', help='see configs\modelnet\modelnet.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config2.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config2.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_logger():
    # 创建一个Logger
    logger = logging.getLogger("mylogger")
    logger.setLevel(logging.INFO)
    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(os.path.join(args.save_path, 'test.log'))
    fh.setLevel(logging.INFO)
    # 定义hander的输出格式（formatter）
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 给handler添加formatter
    fh.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    return logger

def log_string(str):
        logger.info(str)
        print(str)

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in enumerate(tqdm(loader, desc='eval', ncols=0)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct) # insAcc
    return instance_acc, class_acc

def modelnet_evaluate():
    global args, logger
    args = get_parser()
    logger = get_logger()
    log_string(args)
    assert args.num_category > 1
    log_string("=> creating model ...")
    log_string("Classes: {}".format(args.num_category))

    from models.modelnet import Model
    model = Model(num_classes = args.num_category, extra_feature_channels = args.extra_feature_channels)
    model = torch.nn.DataParallel(model.cuda())

    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        log_string("=> loaded checkpoint '{}'".format(args.model_path))
        del checkpoint
    else:
        return
    TEST_DATASET = ModelNetDataset(root=args.data_root, npoint=args.num_point, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.train_batch_size_val, shuffle=False, num_workers=args.train_workers)

    with torch.no_grad():
            instance_acc, class_acc = test(model.eval(), testDataLoader)
    log_string('evaluate Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

