import torch
import os
import numpy as np
from numpy.random import randint
import pandas as pd
import time

def I3D_Pooling(prop_indices, vid, ft_path, n_frame, n_seg, vids=None):
    # ft_tensor = torch.load(os.path.join(ft_path, vid))
    fts_all_act = []
    fts_all_comp = []

    if vids is not None:
        for cnt, prop in enumerate(prop_indices):

            ft_tensor = torch.load(os.path.join(ft_path, vids[cnt]))
            act_s = prop[0]
            act_e = prop[1]
            comp_s = prop[2]
            comp_e = prop[3]

            start_ft = feature_pooling(comp_s, act_s, vid,
                                    n_frame, n_seg, 'max', ft_tensor)
            end_ft = feature_pooling(act_e, comp_e, vid,
                                    n_frame, n_seg, 'max', ft_tensor)
            act_ft = feature_pooling(act_s, act_e, vid,
                                    n_frame, n_seg, 'max', ft_tensor)
            comp_ft = [start_ft, act_ft, end_ft]
            comp_ft = torch.cat(comp_ft, dim=0)

            fts_all_act.append(act_ft)
            fts_all_comp.append(comp_ft)
    else:
        ft_tensor = torch.load(os.path.join(ft_path, vid))
        for cnt, prop in enumerate(prop_indices):

            act_s = prop[0]
            act_e = prop[1]
            comp_s = prop[2]
            comp_e = prop[3]

            start_ft = feature_pooling(comp_s, act_s, vid,
                                    n_frame, n_seg, 'max', ft_tensor)
            end_ft = feature_pooling(act_e, comp_e, vid,
                                    n_frame, n_seg, 'max', ft_tensor)
            act_ft = feature_pooling(act_s, act_e, vid,
                                    n_frame, n_seg, 'max', ft_tensor)
            comp_ft = [start_ft, act_ft, end_ft]
            comp_ft = torch.cat(comp_ft, dim=0)

            fts_all_act.append(act_ft)
            fts_all_comp.append(comp_ft)


    fts_all_act = torch.stack(fts_all_act)
    fts_all_comp = torch.stack(fts_all_comp)

    return fts_all_act, fts_all_comp

def feature_pooling(start_ind, end_ind, vid, n_frame, n_seg, type, ft_tensor):
    #for turn
    interval = 8
    clip_length = 64

    fts_all = []

    offsets, average_duration = sample_indices(start_ind, end_ind, n_seg)

    ft_num = ft_tensor.size()[0]

    for off in offsets:

        start_unit = int(min(ft_num-1, np.floor(float(start_ind+off)*100/n_frame)))
        end_unit = int(min(ft_num-2, np.ceil(float(start_ind+off+average_duration)*100/n_frame)))

        if start_unit < end_unit:
            fts_all.append(torch.max(ft_tensor[start_unit: end_unit+1, :], 0)[0])
        else:
            fts_all.append(ft_tensor[start_unit])

    fts_all = torch.cat(fts_all)

    return fts_all.squeeze()

def sample_indices(start, end, num_seg):
    """
    :param record: VideoRecord
    :return: list
    """
    valid_length = end - start
    average_duration = (valid_length + 1) / num_seg
    offsets = np.multiply(list(range(num_seg)), average_duration)

    return offsets, average_duration
