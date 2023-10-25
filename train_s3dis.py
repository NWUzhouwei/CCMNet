import argparse
import os
import random
import shutil
import numpy as np
# python train_s3dis.py configs\s3dis\model2\area5\c1.py --devices 0,1 --evaluate --visual
seedseed = 1234

def prepare():
    from utils.common import get_save_path
    from utils.config import configs
    from utils.device import set_cuda_visible_devices

    # since PyTorch jams device selection, we have to parse args before import torch (issue #26790)
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--devices', default=None)
    parser.add_argument('--evaluate', default=False, action='store_true')
    parser.add_argument('--visual', default=False, action='store_true')
    args, opts = parser.parse_known_args()
    if args.devices is not None and args.devices != 'cpu':
        gpus = set_cuda_visible_devices(args.devices)
    else:
        gpus = []

    print(f'==> loading configs from {args.configs}')
    configs.update_from_modules(*args.configs)
    configs.visual = args.visual
    # define save path
    # configs.train.save_path = get_save_path('s3dis\\model_' + str(configs.dataset.holdout_area), prefix='runs')
    configs.train.save_path = get_save_path('s3dis', prefix='runs')
    
    # override configs with args
    configs.update_from_arguments(*opts)
    if len(gpus) == 0:
        configs.device = 'cpu'
        configs.device_ids = []
    else:
        configs.device = 'cuda'
        configs.device_ids = gpus
    if args.evaluate and configs.evaluate.fn is not None:
        if 'dataset' in configs.evaluate:
            for k, v in configs.evaluate.dataset.items():
                configs.dataset[k] = v
    else:
        configs.evaluate = None

    if configs.evaluate is None:
        metrics = []
        if 'metric' in configs.train and configs.train.metric is not None:
            metrics.append(configs.train.metric)
        if 'metrics' in configs.train and configs.train.metrics is not None:
            for m in configs.train.metrics:
                if m not in metrics:
                    metrics.append(m)
        configs.train.metrics = metrics
        configs.train.metric = None if len(metrics) == 0 else metrics[0]

        save_path = configs.train.save_path
        configs.train.checkpoint_path = os.path.join(save_path, 'latest.pth.tar')
        configs.train.checkpoints_path = os.path.join(save_path, 'latest', 'e{}.pth.tar')
        configs.train.best_checkpoint_path = os.path.join(configs.train.save_path, 'best.pth.tar')
        best_checkpoints_dir = os.path.join(save_path, 'best')
        configs.train.best_checkpoint_paths = {
            m: os.path.join(best_checkpoints_dir, 'best.{}.pth.tar'.format(m.replace('\\', '.')))
            for m in configs.train.metrics
        }
        os.makedirs(os.path.dirname(configs.train.checkpoints_path), exist_ok=True)
        os.makedirs(best_checkpoints_dir, exist_ok=True)
    else:
        if 'best_checkpoint_path' not in configs.evaluate or configs.evaluate.best_checkpoint_path is None:
            if 'best_checkpoint_path' in configs.train and configs.train.best_checkpoint_path is not None:
                configs.evaluate.best_checkpoint_path = configs.train.best_checkpoint_path
            else:
                configs.evaluate.best_checkpoint_path = os.path.join(configs.train.save_path, 'best.pth.tar')
        assert configs.evaluate.best_checkpoint_path.endswith('.pth.tar')
        configs.evaluate.predictions_path = configs.evaluate.best_checkpoint_path.replace('.pth.tar', '.predictions')
        configs.evaluate.stats_path = configs.evaluate.best_checkpoint_path.replace('.pth.tar', '.eval.npy')

    return configs

def worker_init_fn(worker_id):
    random.seed(seedseed + worker_id)


def main():
    configs = prepare()
    if configs.evaluate is not None:
        configs.evaluate.fn(configs)
        return

    
    import tensorboardX
    import torch
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import time

    ################################
    # Train / Eval Kernel Function #
    ################################

    # train kernel
    def train(model, loader, criterion, optimizer, scheduler, current_step, split='train'):
        model.train()
        meters = {}
        for k, meter in configs.train.meters.items(): # key value
            meters[k.format(split)] = meter()
        train_loss = 0.0
        count = 0
        for inputs, targets in tqdm(loader, desc='train', ncols=0):
            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    batch_size = v.size(0)
                    inputs[k] = v.to(configs.device, non_blocking=True)
            else:
                batch_size = inputs.size(0)
                inputs = inputs.to(configs.device, non_blocking=True)
            if isinstance(targets, dict):
                for k, v in targets.items():
                    targets[k] = v.to(configs.device, non_blocking=True)
            else:
                targets = targets.to(configs.device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            current_step += batch_size
            count += batch_size
            train_loss += loss.item() * batch_size
            loss.backward()
            optimizer.step()
            for meter in meters.values():
                    meter.update(outputs, targets)
        for k, meter in meters.items():
            meters[k] = meter.compute()
        if scheduler is not None:
            scheduler.step()
        return meters, train_loss*1.0/count

    # evaluate kernel
    def evaluate(model, loader, current_step, split='test'):
        meters = {}
        test_loss = 0.0
        count = 0
        for k, meter in configs.train.meters.items(): # key value
            meters[k.format(split)] = meter()
        model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=split, ncols=0):
                if isinstance(inputs, dict):
                    for k, v in inputs.items():
                        batch_size = v.size(0)
                        inputs[k] = v.to(configs.device, non_blocking=True)
                else:
                    batch_size = inputs.size(0)
                    inputs = inputs.to(configs.device, non_blocking=True)
                if isinstance(targets, dict):
                    for k, v in targets.items():
                        targets[k] = v.to(configs.device, non_blocking=True)
                else:
                    targets = targets.to(configs.device, non_blocking=True)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                current_step += batch_size
                count += batch_size
                test_loss += loss.item() * batch_size
                
                for meter in meters.values():
                    meter.update(outputs, targets)
        for k, meter in meters.items():
            meters[k] = meter.compute()
        return meters, test_loss*1.0/count

    ###########
    # Prepare #
    ###########
    if configs.device == 'cuda':
        cudnn.benchmark = True
        if configs.get('deterministic', False):
            cudnn.deterministic = True
            cudnn.benchmark = False
    if ('seed' not in configs) or (configs.seed is None):
        configs.seed = torch.initial_seed() % (2 ** 32 - 1)
    seed = configs.seed
    seedseed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # print(configs)

    import logging
    def log_string(str):
        logger.info(str)
        print(str)

    logger = logging.getLogger("mylogger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(configs.train.save_path, 'train.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    log_string('PARAMETER ...')
    log_string(configs)

    #####################################################################
    # Initialize DataLoaders, Model, Criterion, LRScheduler & Optimizer #
    #####################################################################

    log_string(f'\n==> loading dataset "{configs.dataset}"')
    dataset = configs.dataset()
    loaders = {}
    for split in dataset:
        loaders[split] = DataLoader(
            dataset[split], shuffle=(split == 'train'), batch_size=configs.train.batch_size,
            num_workers=configs.data.num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn
        )
    log_string("The number of train data is: %d" % len(dataset['train']))
    log_string("The number of test data is: %d" % len(dataset['test']))


    log_string(f'\n==> creating model "{configs.model}"')
    model = configs.model()
    if configs.device == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(configs.device)
    criterion = configs.train.criterion().to(configs.device)
    optimizer = configs.train.optimizer(model.parameters())

    last_epoch, best_metrics = -1, {m: None for m in configs.train.metrics}
    if os.path.exists(configs.train.checkpoint_path):
        log_string(f'==> loading checkpoint "{configs.train.checkpoint_path}"')
        checkpoint = torch.load(configs.train.checkpoint_path)
        log_string(' => loading model')
        model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            log_string(' => loading optimizer')
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        last_epoch = checkpoint.get('epoch', last_epoch)
        meters = checkpoint.get('meters', {})
        for m in configs.train.metrics:
            best_metrics[m] = meters.get(m + '_best', best_metrics[m])
        del checkpoint

    if 'scheduler' in configs.train and configs.train.scheduler is not None:
        configs.train.scheduler.last_epoch = last_epoch
        log_string(f'==> creating scheduler "{configs.train.scheduler}"')
        scheduler = configs.train.scheduler(optimizer)
    else:
        scheduler = None

    ############
    # Training #
    ############

    if last_epoch >= configs.train.num_epochs:
        meters = dict()
        for split, loader in loaders.items():
            if split != 'train':
                meters.update(evaluate(model, loader=loader, split=split))
        for k, meter in meters.items():
            log_string(f'[{k}] = {meter:2f}')
        return

    with tensorboardX.SummaryWriter(configs.train.save_path) as writer:
        for current_epoch in range(last_epoch + 1, configs.train.num_epochs):
            current_step = current_epoch * len(dataset['train'])

            log_string(f'\n==> training epoch {current_epoch}/{configs.train.num_epochs}')
            # train
            meters1 = dict()
            since = time.time() # add
            b, train_loss = train(model, loader=loaders['train'], criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                  current_step=current_step)

            ttt = time.time() - since
            log_string('Training complete in {:.0f}m {:.0f}s'.format(ttt // 60, ttt % 60))
            meters1.update(b)
            
            for k, meter in meters1.items():
                log_string(f'[{k}] = {meter:2f}') # 打印iou
                writer.add_scalar(k, meter, current_epoch)
            log_string(f'Train loss : {train_loss:2f}')
            # evaluate
            meters2 = dict()
            for split, loader in loaders.items():
                if split != 'train':
                    since = time.time()
                    a, test_loss= evaluate(model, loader=loader, current_step=current_step, split=split)
                    ttt = time.time() - since
                    log_string('Testing complete in {:.0f}m {:.0f}s'.format(ttt // 60, ttt % 60))
                    meters2.update(a)

            current_step += len(dataset['train'])

            writer.add_scalars('loss', {'train' : train_loss, 'test' : test_loss}, current_epoch)
            

            # check whether it is the best
            best = {m: False for m in configs.train.metrics}
            for m in configs.train.metrics:
                if best_metrics[m] is None or best_metrics[m] < meters2[m]:
                    best_metrics[m], best[m] = meters2[m], True
                meters2[m + '_best'] = best_metrics[m]
            for k, meter in meters2.items():
                log_string(f'[{k}] = {meter:2f}') # 打印iou
                writer.add_scalar(k, meter, current_epoch)
            log_string(f'Test loss : {test_loss:2f}')

            # save checkpoint
            torch.save({
                'epoch': current_epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'meters': meters2,
                'configs': configs,
            }, configs.train.checkpoint_path)
            
            for m in configs.train.metrics:
                if best[m]:
                    shutil.copyfile(configs.train.checkpoint_path, configs.train.best_checkpoint_paths[m])
            if best.get(configs.train.metric, False):
                shutil.copyfile(configs.train.checkpoint_path, configs.train.best_checkpoint_path)
            log_string(f'[save_path] = {configs.train.save_path}')


if __name__ == '__main__':
    main()