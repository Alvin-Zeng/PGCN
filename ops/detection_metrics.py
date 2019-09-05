"""
This module provides some utils for calculating metrics in temporal action detection
"""
import numpy as np


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

def segment_tiou(target_segments, test_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    t_overlap = np.empty((m, n))
    tiou = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) +
                 (target_segments[i, 1] - target_segments[i, 0] + 1) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
        t_overlap[i, :] = intersection
    return tiou, t_overlap

def segment_distance(target_segments, test_segments):
    """Compute distance btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with distance.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    target_centers = (target_segments[:, 1] - target_segments[:, 0])/2 + target_segments[:, 0]
    test_centers = (test_segments[:, 1] - test_segments[:, 0])/2 + test_segments[:, 0]
    dis_mat = np.empty((m, n))

    for i in range(m):
        target_center = target_centers[i]
        distance = np.abs(test_centers - target_center)

        tt1 = np.maximum(target_segments[i, 1], test_segments[:, 1])
        tt2 = np.minimum(target_segments[i, 0], test_segments[:, 0])
        # Non-negative overlap score
        union = (tt1 - tt2 + 1.0).clip(0)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        dis_mat[i, :] = distance / union

    return dis_mat


def overlap_over_b(span_A, span_B):
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])
    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(span_B[1] - span_B[0])


def temporal_recall(gt_spans, est_spans, thresh=0.5):
    """
    Calculate temporal recall of boxes and estimated boxes
    Parameters
    ----------
    gt_spans: [(start, end), ...]
    est_spans: [(start, end), ...]

    Returns
    recall_info: (hit, total)
    -------

    """
    hit_slot = [False] * len(gt_spans)
    for i, gs in enumerate(gt_spans):
        for es in est_spans:
            if temporal_iou(gs, es) > thresh:
                hit_slot[i] = True
                break
    recall_info = (np.sum(hit_slot), len(hit_slot))
    return recall_info


def name_proposal(gt_spans, est_spans, thresh=0.0):
    """
    Assigng label to positive proposals
    :param gt_spans: [(label, (start, end)), ...]
    :param est_spans: [(start, end), ...]
    :param thresh:
    :return: [(label, overlap, start, end), ...] same number of est_spans
    """
    ret = []
    for es in est_spans:
        max_overlap = 0
        max_overlap_over_self = 0
        label = 0
        for gs in gt_spans:
            ov = temporal_iou(gs[1], es)
            ov_pr = overlap_over_b(gs[1], es)
            if ov > thresh and ov > max_overlap:
                label = gs[0] + 1
                max_overlap = ov
                max_overlap_over_self = ov_pr
        ret.append((label, max_overlap, max_overlap_over_self, es[0], es[1]))

    return ret


def get_temporal_proposal_recall(pr_list, gt_list, thresh):
    recall_info_list = [temporal_recall(x, y, thresh=thresh) for x, y in zip(gt_list, pr_list)]
    per_video_recall = np.sum([x[0] == x[1] for x in recall_info_list]) / float(len(recall_info_list))
    per_inst_recall = np.sum([x[0] for x in recall_info_list]) / float(np.sum([x[1] for x in recall_info_list]))
    return per_video_recall, per_inst_recall

