import argparse
import os
import torch
import numpy as np
from pgcn_dataset import PGCNDataSet
from pgcn_models import PGCN
from torch import multiprocessing
from torch.utils import model_zoo
from ops.utils import get_configs
from ops.I3D_Pooling import I3D_Pooling
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(
    description="PGCN Testing Tool")
parser.add_argument('dataset', type=str, choices=['activitynet1.2', 'thumos14'])
parser.add_argument('weights', type=str)
parser.add_argument('save_scores', type=str)

parser.add_argument('--save_raw_scores', type=str, default=None)
parser.add_argument('--no_regression', action="store_true", default=False)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)

SEED = 777
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,2,3,4,5,6,7"

configs = get_configs(args.dataset)
dataset_configs = configs['dataset_configs']
model_configs = configs["model_configs"]
graph_configs = configs["graph_configs"]

adj_num = graph_configs['adj_num']
num_class = model_configs['num_class']

gpu_list = args.gpus if args.gpus is not None else range(8)



def sample_child_nodes(center_prop_cnt, iou_num, dis_num, child_num, iou_dict, dis_dict):
    # obtain iou array for all the proposals

    act_iou_array = iou_dict[center_prop_cnt, :]
    act_iou_array = np.squeeze(act_iou_array)

    sorted_iou_idx = np.argsort(-act_iou_array).tolist()
    max_iou_ind = sorted_iou_idx[0]
    # remove self
    rm_act_iou_array = act_iou_array.copy()
    rm_act_iou_array[max_iou_ind] = 0
    # filter the proposals
    pos_iou_idx = np.where(rm_act_iou_array > dataset_configs['iou_threshold'])[0]
    if pos_iou_idx.size != 0:
        pos_iou_arr = rm_act_iou_array[pos_iou_idx]
        sorted_pos_iou_idx = np.argsort(-pos_iou_arr).tolist()
        selected_pos_iou_idx = np.tile(sorted_pos_iou_idx, iou_num)
        ref_iou_idx = selected_pos_iou_idx[:iou_num]
        abs_iou_idx = pos_iou_idx[ref_iou_idx]
    else:
        abs_iou_idx = np.tile(np.array(max_iou_ind), iou_num)

    # obtain dis array for all the proposals
    act_dis_array = dis_dict[center_prop_cnt, :]
    act_dis_array = np.squeeze(act_dis_array)

    selected_ious_ind = act_iou_array <= 0
    selected_dis_ind = act_dis_array > dataset_configs['dis_threshold']
    selected_ind = np.logical_and(selected_ious_ind, selected_dis_ind)
    pos_dis_idx = np.where(selected_ind == 1)[0]
    if pos_dis_idx.size != 0:
        pos_dis_arr = act_dis_array[pos_dis_idx]
        sorted_pos_dis_idx = np.argsort(pos_dis_arr).tolist()
        selected_pos_dis_idx = np.tile(sorted_pos_dis_idx, dis_num)
        ref_dis_idx = selected_pos_dis_idx[:dis_num]
        abs_dis_idx = pos_dis_idx[ref_dis_idx]
    else:
        abs_dis_idx = np.tile(np.array(max_iou_ind), dis_num)

    # obtain child idxs
    abs_child_idx = np.concatenate([abs_iou_idx, abs_dis_idx])
    np.random.shuffle(abs_child_idx)
    abs_child_idx = abs_child_idx[:child_num]

    return [ind for ind in abs_child_idx]


def get_adjacent_batch(prop_idx, iou_dict, dis_dict):

    selected_idx = [prop_idx]

    for stage_cnt in range(graph_configs['child_num'] + 1):
        # sample proposal with the largest iou
        idxs = sample_child_nodes(selected_idx[stage_cnt],
                                  graph_configs['iou_num'],
                                  graph_configs['dis_num'],
                                  graph_configs['child_num'],
                                  iou_dict, dis_dict)
        selected_idx.extend(idxs)

    return selected_idx


def runner_func(dataset, state_dict, stats, gpu_id, index_queue, result_queue, iou_dict, dis_dict):

    torch.cuda.set_device(gpu_id)
    net = PGCN(model_configs, graph_configs, test_mode=True)
    net.load_state_dict(state_dict)
    # net.prepare_test_fc()
    net.eval()
    net.cuda()

    while True:
        index = index_queue.get()
        
        rel_props, prop_ticks, video_id, n_frames = dataset[index]

        # calculate scores
        n_out = prop_ticks.size(0)
        act_scores = torch.zeros((n_out, num_class + 1)).cuda()
        comp_scores = torch.zeros((n_out, num_class)).cuda()

        if not args.no_regression:
            reg_scores = torch.zeros((n_out, num_class * 2)).cuda()
        else:
            reg_scores = None

        # load prop fts
        vid_full_name = video_id
        vid = vid_full_name.split('/')[-1]

        act_all_fts, comp_all_fts = I3D_Pooling(prop_ticks, vid, dataset_configs['test_ft_path'], n_frames)

        for prop_idx, prop in enumerate(prop_ticks):
            if prop_idx >= n_out:
                break
            with torch.no_grad():

                vid_iou_dict = iou_dict[vid]
                vid_dis_dict = dis_dict[vid]
                # print(len(vid_iou_dict), len(vid_dis_dict), len(prop_ticks))
                selected_idx= get_adjacent_batch(prop_idx, vid_iou_dict, vid_dis_dict)
                selected_idx = torch.from_numpy(np.array(selected_idx))
                act_ft = act_all_fts[selected_idx, :]
                comp_ft = comp_all_fts[selected_idx, :]

                act_batch_var = act_ft.unsqueeze(0).cuda()
                comp_batch_var = comp_ft.unsqueeze(0).cuda()

                act_scores[prop_idx, :], comp_scores[prop_idx, :], \
                reg_scores[prop_idx, :] = net((act_batch_var, comp_batch_var), None, None, None)

        if reg_scores is not None:
            reg_scores = reg_scores.view(-1, num_class, 2)
            reg_scores[:, :, 0] = reg_scores[:, :, 0] * stats[1, 0] + stats[0, 0]
            reg_scores[:, :, 1] = reg_scores[:, :, 1] * stats[1, 1] + stats[0, 1]

        # perform stpp on scores
        result_queue.put((dataset.video_list[index].id, rel_props.numpy(), act_scores.cpu().numpy(),
                          comp_scores.cpu().numpy(), reg_scores.cpu().numpy(), 0))


if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')  # this is crucial to using multiprocessing processes with PyTorch


    # This net is used to provides setup settings. It is not used for testing.

    checkpoint = torch.load(args.weights)

    print("model epoch {} loss: {}".format(checkpoint['epoch'], checkpoint['best_loss']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    stats = checkpoint['reg_stats'].numpy()

    dataset = PGCNDataSet(dataset_configs, graph_configs,
                          prop_file=dataset_configs['test_prop_file'],
                          prop_dict_path=dataset_configs['test_dict_path'],
                          ft_path=dataset_configs['test_ft_path'],
                          test_mode=True)

    iou_dict = dataset.act_iou_dict
    dis_dict = dataset.act_dis_dict

    index_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=runner_func,
                           args=(dataset, base_dict, stats, gpu_list[i % len(gpu_list)],
                                 index_queue, result_queue,  iou_dict,
                                 dis_dict))
               for i in range(args.workers)]


    for w in workers:
        w.daemon = True
        w.start()

    max_num = args.max_num if args.max_num > 0 else len(dataset)

    for i in range(max_num):
        index_queue.put(i)

    out_dict = {}
    pbar = tqdm(total=max_num)
    for i in range(max_num):
        pbar.update(1)
        rst = result_queue.get()
        out_dict[rst[0]] = rst[1:]
    pbar.close()

    if args.save_scores is not None:
        save_dict = {k: v[:-1] for k, v in out_dict.items()}
        import pickle

        pickle.dump(save_dict, open(args.save_scores, 'wb'), pickle.HIGHEST_PROTOCOL)

    if args.save_raw_scores is not None:
        save_dict = {k: v[-1] for k, v in out_dict.items()}
        import pickle

        pickle.dump(save_dict, open(args.save_raw_scores, 'wb'), pickle.HIGHEST_PROTOCOL)
