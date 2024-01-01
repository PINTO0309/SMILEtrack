from __future__ import annotations
import numpy as np
from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter
from tracker.matching import STrack
from fast_reid.fast_reid_interfece import FastReIDInterface
from typing import List, Dict

class BoTSORT(object):
    def __init__(self, args, frame_rate: int=30):

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh: float = args.track_high_thresh # tracking confidence threshold 0.6
        self.track_low_thresh: float = args.track_low_thresh # lowest detection threshold valid for tracks 0.1
        self.new_track_thresh: float = args.new_track_thresh # new track thresh 0.7

        self.buffer_size: int = int(frame_rate / 30.0 * args.track_buffer) # the frames for keep lost tracks 30
        self.max_time_lost: int = self.buffer_size
        self.kalman_filter: KalmanFilter = KalmanFilter()

        # ReID module
        self.proximity_thresh: float = args.proximity_thresh # threshold for rejecting low overlap reid matches 0.5
        self.appearance_thresh: float = args.appearance_thresh # threshold for rejecting low appearance similarity reid matches 0.25

        if args.with_reid:
            # args.fast_reid_config: fast_reid/configs/MOT17/sbs_S50.yml
            # args.fast_reid_weights: pretrained/mot17_sbs_S50.pth
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        # args.cmc_method: "file"
        # args.name: "MOT17-01-FRCNN"
        # args.ablation: False
        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

    def update(self, output_results: np.ndarray, img: np.ndarray):
        self.frame_id += 1
        activated_starcks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores: np.ndarray = output_results[:, 4]
                bboxes: np.ndarray = output_results[:, :4]
                classes: np.ndarray = output_results[:, -1]
            else:
                scores: np.ndarray = output_results[:, 4] * output_results[:, 5]
                bboxes: np.ndarray = output_results[:, :4]  # x1y1x2y2
                classes: np.ndarray = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes: np.ndarray = bboxes[lowest_inds]
            scores: np.ndarray = scores[lowest_inds]
            classes: np.ndarray = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.track_high_thresh
            dets: np.ndarray = bboxes[remain_inds]
            scores_keep: np.ndarray = scores[remain_inds]
            classes_keep: np.ndarray = classes[remain_inds]

        else:
            bboxes = np.asarray([])
            scores = np.asarray([])
            classes = np.asarray([])
            dets = np.asarray([])
            scores_keep = np.asarray([])
            classes_keep = np.asarray([])

        '''Extract embeddings '''
        if self.args.with_reid:
            # img: [H, W, 3]
            # dets: [N, 4] -> [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
            # features_keep: [N, 2048]
            features_keep: np.ndarray = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections: List[STrack] = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track: STrack = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = np.asarray([])
            scores_second = np.asarray([])
            classes_second = np.asarray([])

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second: List[STrack] = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second: List[STrack] = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed_track: STrack = unconfirmed[itracked]
            unconfirmed_track.update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed_track)
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks: List[STrack] = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks: List[STrack] = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks: List[STrack] = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks: List[STrack] = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks


def joint_stracks(tlista: List[STrack], tlistb: List[STrack]):
    exists: Dict[int, int] = {}
    res: List[STrack] = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista: List[STrack], tlistb: List[STrack]):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa: List[STrack], stracksb: List[STrack]):
    pdist: np.ndarray = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        strackp: STrack =stracksa[p]
        timep = strackp.frame_id - strackp.start_frame
        strackq: STrack =stracksb[q]
        timeq = strackq.frame_id - strackq.start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
