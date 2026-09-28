import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from shar.schemas import ActorTrack


def box_iou(a, b):
    intersection = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x)) * max(
        0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    )
    return intersection / (a.width * a.height + b.width * b.height - intersection)


def associate(detections, max_gap=1.0, min_iou=0.2, max_distance=0.15):
    tracks: list[ActorTrack] = []
    for timestamp in sorted({d.timestamp_sec for d in detections}):
        observations = [d for d in detections if d.timestamp_sec == timestamp]
        active = [t for t in tracks if 0 < timestamp - t.samples[-1].timestamp_sec <= max_gap]
        costs = np.full((len(active), len(observations)), 1e6)
        for i, track in enumerate(active):
            a = track.samples[-1].bbox
            for j, detection in enumerate(observations):
                b = detection.bbox
                overlap = box_iou(a, b)
                distance = math.hypot(
                    a.x + a.width / 2 - b.x - b.width / 2, a.y + a.height / 2 - b.y - b.height / 2
                )
                if overlap >= min_iou or distance <= max_distance:
                    costs[i, j] = 1 - overlap + distance
        matched, ambiguous = set(), set()
        if costs.size:
            for i, j in zip(*linear_sum_assignment(costs)):
                if costs[i, j] >= 1e6:
                    continue
                row, col = sorted(costs[i]), sorted(costs[:, j])
                if (len(row) > 1 and row[1] - row[0] < 0.05) or (
                    len(col) > 1 and col[1] - col[0] < 0.05
                ):
                    active[i].association_status = "ambiguous"
                    ambiguous.add(j)
                    continue
                active[i].samples.append(observations[j])
                matched.add(j)
        for j, observation in enumerate(observations):
            if j not in matched:
                tracks.append(
                    ActorTrack(
                        samples=[observation],
                        association_status="ambiguous" if j in ambiguous else "provisional",
                    )
                )
    return tracks
