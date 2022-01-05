#build_detector 函数将配置文件中的：model、train_cfg 和 test_cfg 传入参数。
#运行时会将上面的三个值作为参数传入 build_detector 函数，
    #build_detector 函数会调用 build 函数
    #build 函数调用 build_from_cfg 函数构建检测器对象
    #其中 train_cfg 和 test_cfg 作为默认参数用于构建 detector 对象。
from mmcv.utils import Registry, build_from_cfg
from torch import nn

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        # 调用 build_from_cfg 用来根据 config 字典构建 registry 里面的对象
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    # 调用 build 函数，传入 cfg, registry 对象，
    # 把 train_cfg 和 test_cfg 作为默认字典传入
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
