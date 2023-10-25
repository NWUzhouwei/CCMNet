from datasets import ModelNetDataset
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import provider
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from utils import config2
from evaluate.modelnet.eval import modelnet_evaluate as eval_code
# python train_modelnet.py --config configs\\modelnet\\modelnet.yaml
def get_parser():
    parser = argparse.ArgumentParser(description='ModelNet Dataset Instances')
    parser.add_argument('--config', type=str, default='--config configs\\modelnet\\modelnet.yaml', help='config file')
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
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    fh = logging.FileHandler(os.path.join(args.save_path, 'train.log'))
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

def init():
    global args, logger, writer
    args = get_parser()
    if args.test is True:
        eval_code()
        exit()
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    log_string('PARAMETER ...')
    log_string(args)
    

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

def main():
    init()
    from models.modelnet import Model
    model = Model(num_classes = args.num_category, extra_feature_channels = args.extra_feature_channels)

    TRAIN_DATASET = ModelNetDataset(root=args.data_root, npoint=args.num_point, split='train')
    TEST_DATASET = ModelNetDataset(root=args.data_root, npoint=args.num_point, split='test')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_workers)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.train_batch_size_val, shuffle=False, num_workers=args.train_workers)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
    log_string("Classes: {}".format(args.num_category))
    model = torch.nn.DataParallel(model.cuda())
    if args.resume:
        if os.path.isfile(args.resume):
            log_string("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            log_string("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            log_string("=> no checkpoint found at '{}'".format(args.resume))

    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    if args.best_path:
        if os.path.isfile(args.best_path):
            log_string("=> loading checkpoint '{}'".format(args.best_path))
            checkpoint = torch.load(args.best_path, map_location=lambda storage, loc: storage.cuda())
            best_instance_acc = checkpoint['best_instance_acc']

    for epoch in range(args.start_epoch, args.epochs):
        log_string('\n==> training epoch [{}/{}]:'.format(epoch+1, args.epochs))
        model.train()
        for batch_id, data in enumerate(tqdm(trainDataLoader, desc='train', ncols=0)):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            pred = model(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        writer.add_scalar('InsAcc_train', train_instance_acc, epoch+1)

        

        # evaluate
        with torch.no_grad():
            instance_acc, class_acc = test(model.eval(), testDataLoader)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
                torch.save({'epoch': best_epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'best_instance_acc': best_instance_acc}, args.best_path)
                log_string('==>Saving best checkpoint to: ' + args.best_path)
            
            log_string('==>Saving checkpoint to: ' + args.resume)
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, args.resume)

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc

            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
            
            writer.add_scalar('InsAcc_test', instance_acc, epoch+1)
            writer.add_scalar('bestInsAcc', best_instance_acc, epoch+1)
            writer.add_scalar('ClassAcc_test', class_acc, epoch+1)
            writer.add_scalar('bestClassAcc', best_class_acc, epoch+1)

    log_string('End of training...')

if __name__ == '__main__':
    main()