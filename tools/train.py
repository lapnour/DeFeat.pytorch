import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
# Config 用于读取配置文件, DictAction 将命令行字典类型参数转化为 key-value 形式
from mmcv import Config, DictAction
from mmcv.runner import init_dist

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

# python tools/train.py ${CONFIG_FILE} [optional arguments]


# --work-dir        存储日志和模型的目录
# --resume-from     加载 checkpoint 的目录
# --no-validate     是否在训练的时候进行验证
# 互斥组：
#   --gpus          使用的 GPU 数量
#   --gpu_ids       使用指定 GPU 的 id
# --seed            随机数种子
# --deterministic   是否设置 cudnn 为确定性行为
# --options         其他参数
# --launcher        分布式训练使用的启动器，可以为：['none', 'pytorch', 'slurm', 'mpi']
#                   none：不启动分布式训练，dist_train.sh 中默认使用 pytorch 启动。
# --local_rank      本地进程编号，此参数 torch.distributed.launch 会自动传入。

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    
# action: store (默认, 表示保存参数)
# action: store_true, store_false (如果指定参数, 则为 True, False)

    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    
# --------- 创建一个互斥组. argparse 将会确保互斥组中的参数只能出现一个 ---------    
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    
# 可以使用 python train.py --gpu-ids 0 1 2 3 指定使用的 GPU id
    # 参数结果：[0, 1, 2, 3]
    # nargs = '*'：参数个数可以设置0个或n个
    # nargs = '+'：参数个数可以设置1个或n个
    # nargs = '?'：参数个数可以设置0个或1个  
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    
# 如果使用 dist_utils.sh 进行分布式训练, launcher 默认为 pytorch
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args, unparsed = parser.parse_known_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    # Solve SyncBN deadlock
    os.environ["NCCL_LL_THRESHOLD"] = '0'

    return args


def main():
    args = parse_args()
    
#从命令行和配置文件获取参数配置

    #从文件读取配置
    cfg = Config.fromfile(args.config)
    #从命令行读取
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # 设置 cudnn_benchmark = True 可以加速输入大小固定的模型. 如：SSD300？
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename（命令行>配置）
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    
# 构建模型: 需要传入 cfg.model，cfg.train_cfg，cfg.test_cfg
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    from mmcv.runner import get_dist_info
    rank, world_size = get_dist_info()
    if rank == 0:
        print(model)
        print("Model have {} paramerters.".format(sum(x.numel() for x in model.parameters()) / 1e6))
        if hasattr(model, 'backbone'):
            print("Model has {} backbone.".format(sum(x.numel() for x in model.backbone.parameters()) / 1e6))
        if hasattr(model, 'neck'):
            print("Model has {} neck.".format(sum(x.numel() for x in model.neck.parameters()) / 1e6))
        if hasattr(model, 'roi_head'):
            print("Model has {} bbox head.".format(sum(x.numel() for x in model.roi_head.bbox_head.parameters()) / 1e6))
        if hasattr(model, 'bbox_head'):
            print("Model has {} bbox head.".format(sum(x.numel() for x in model.bbox_head.parameters()) / 1e6))
            
# 构建数据集: 需要传入 cfg.data.train，表明是训练集
    datasets = [build_dataset(cfg.data.train)]
    # workflow 代表流程：
    # [('train', 2), ('val', 1)] 就代表，训练两个 epoch 验证一个 epoch
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    
# 训练检测器：需要传入模型、数据集、配置参数等
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
