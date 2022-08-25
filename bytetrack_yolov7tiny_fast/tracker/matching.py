# import cv2
# import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from .kalman_filter import *
# import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    # atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float)
    # btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float)
    # ious = max([iou(atlbrs[i], btlbrs[i]) for i in range(len(atlbrs))])
    ious = bbox_ious(np.ascontiguousarray(atlbrs, dtype=np.float), np.ascontiguousarray(btlbrs, dtype=np.float))
    return ious

def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def iou(box1, box2) -> float:
    """
    IOU, Intersection over Union

    :param box1: list, 第一个框的两个坐标点位置 box1[x1, y1, x2, y2]
    :param box2: list, 第二个框的两个坐标点位置 box2[x1, y1, x2, y2]
    :return: float, 交并比
    """
    weight = max(min(box1[2], box2[2]) - max(box1[0], box2[0]), 0)
    height = max(min(box1[3], box2[3]) - max(box1[1], box1[1]), 0)
    s_inter = weight * height
    s_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    s_union = s_box1 + s_box2 - s_inter
    return s_inter / s_union

# def iou(bbox, candidates):
#     """Computer intersection over union.
#
#     Parameters
#     ----------
#     bbox : ndarray
#         A bounding box in format `(top left x, top left y, width, height)`.
#     candidates : ndarray
#         A matrix of candidate bounding boxes (one per row) in the same format
#         as `bbox`.
#
#     Returns
#     -------
#     ndarray
#         The intersection over union in [0, 1] between the `bbox` and each
#         candidate. A higher score means a larger fraction of the `bbox` is
#         occluded by the candidate.
#
#     """
#     bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
#     candidates_tl = candidates[:, :2]
#     candidates_br = candidates[:, :2] + candidates[:, 2:]
#
#     tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
#                np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
#     br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
#                np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
#     wh = np.maximum(0., br - tl)
#
#     area_intersection = wh.prod(axis=1)
#     area_bbox = bbox[2:].prod()
#     area_candidates = candidates[:, 2:].prod(axis=1)
#     return area_intersection / (area_bbox + area_candidates - area_intersection)
#
#
# def iou_distance(tracks, detections, track_indices=None,
#              detection_indices=None):
#     """An intersection over union distance metric.
#
#     Parameters
#     ----------
#     tracks : List[deep_sort.track.Track]
#         A list of tracks.
#     detections : List[deep_sort.detection.Detection]
#         A list of detections.
#     track_indices : Optional[List[int]]
#         A list of indices to tracks that should be matched. Defaults to
#         all `tracks`.
#     detection_indices : Optional[List[int]]
#         A list of indices to detections that should be matched. Defaults
#         to all `detections`.
#
#     Returns
#     -------
#     ndarray
#         Returns a cost matrix of shape
#         len(track_indices), len(detection_indices) where entry (i, j) is
#         `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
#
#     """
#     if track_indices is None:
#         track_indices = np.arange(len(tracks))
#     if detection_indices is None:
#         detection_indices = np.arange(len(detections))
#
#     cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
#     for row, track_idx in enumerate(track_indices):
#         if tracks[track_idx].time_since_update > 1:
#             cost_matrix[row, :] = linear_assignment.INFTY_COST
#             continue
#
#         bbox = tracks[track_idx].to_tlwh()
#         candidates = np.asarray(
#             [detections[i].tlwh for i in detection_indices])
#         cost_matrix[row, :] = 1. - iou(bbox, candidates)
#     return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost