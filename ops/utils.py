import torch
import numpy as np
import time
import logging
from ruamel import yaml

def get_logger(args):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)

    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = args.snapshot_pref+date+'.log'
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def get_configs(dataset):
    data = yaml.load(open('data/dataset_cfg.yaml', 'r'), Loader=yaml.RoundTripLoader)
    return data[dataset]

def get_and_save_args(parser):
    args = parser.parse_args()
    dataset = args.dataset

    default_config = yaml.load(open('./data/dataset_cfg.yaml', 'r'), Loader=yaml.RoundTripLoader)[dataset]
    current_config = vars(args)
    for k, v in current_config.items():
        if k in default_config:
            if (v != default_config[k]) and (v is not None):
                print(f"Updating:  {k}: {default_config[k]} (default) ----> {v}")
                default_config[k] = v
    yaml.dump(default_config, open('./current_configs.yaml', 'w'), indent=4, Dumper=yaml.RoundTripDumper)
    return default_config


def get_grad_hook(name):
    def hook(m, grad_in, grad_out):
        print(len(grad_in), len(grad_out))
        print((name, grad_out[0].data.abs().mean(), grad_in[0].data.abs().mean()))
        print((grad_out[0].size()))
        print((grad_in[0].size()))
        print((grad_in[1].size()))
        print((grad_in[2].size()))

        # print((grad_out[0]))
        # print((grad_in[0]))

    return hook


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


def temporal_iou(span_A, span_B):
    """
    Calculates the intersection over union of two temporal "bounding boxes"

    span_A: (start, end)
    span_B: (start, end)
    """
    union = min(span_A[0], span_B[0]), max(span_A[1], span_B[1])
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])

    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(union[1] - union[0])


def temporal_nms(bboxes, thresh):
    """
    One-dimensional non-maximal suppression
    :param bboxes: [[st, ed, score, ...], ...]
    :param thresh:
    :return:
    """
    t1 = bboxes[:, 0]
    t2 = bboxes[:, 1]
    scores = bboxes[:, 2]

    durations = t2 - t1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]

    return bboxes[keep, :]
