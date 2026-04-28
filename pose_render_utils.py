from __future__ import annotations

import cv2
import numpy as np


def visualize_keypoints(
    image: np.ndarray,
    keypoints,
    keypoints_visible,
    keypoint_scores,
    *,
    radius: int = 4,
    thickness: int = -1,
    color=(255, 0, 0),
    kpt_thr: float = 0.3,
    skeleton: list | None = None,
    kpt_color: list | tuple | np.ndarray | None = None,
    link_color: list | tuple | np.ndarray | None = None,
    show_kpt_idx: bool = False,
) -> np.ndarray:
    image = image.copy()
    image_h, image_w = image.shape[:2]

    if skeleton is None:
        skeleton = []
    if kpt_color is None:
        kpt_color = color
    if link_color is None:
        link_color = (0, 255, 0)

    def _as_color_list(value, count: int):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.ndim == 2 and value.shape[1] == 3:
                return [tuple(int(channel) for channel in row) for row in value.tolist()]
            if value.size == 3:
                triplet = tuple(int(channel) for channel in value.reshape(-1).tolist())
                return [triplet] * max(1, count)
        if isinstance(value, (list, tuple)):
            if count and len(value) == count and isinstance(value[0], (list, tuple, np.ndarray)):
                output = []
                for entry in value:
                    triplet = np.asarray(entry).reshape(-1)
                    if triplet.size != 3:
                        raise ValueError("Each keypoint color must be an RGB triplet.")
                    output.append(tuple(int(channel) for channel in triplet.tolist()))
                return output
            triplet = np.asarray(value).reshape(-1)
            if triplet.size == 3:
                color_value = tuple(int(channel) for channel in triplet.tolist())
                return [color_value] * max(1, count)
        return [(255, 0, 0)] * max(1, count)

    def _in_bounds(x: int, y: int) -> bool:
        return 0 <= x < image_w and 0 <= y < image_h

    num_keypoints = keypoints[0].shape[0] if keypoints else 0
    keypoint_colors = _as_color_list(kpt_color, num_keypoints)
    link_colors = _as_color_list(link_color, len(skeleton))

    for keypoints_one, visible_one, scores_one in zip(
        keypoints,
        keypoints_visible,
        keypoint_scores,
    ):
        keypoints_array = np.asarray(keypoints_one, dtype=np.float32)
        visible_array = np.asarray(visible_one).reshape(-1).astype(bool)
        scores_array = np.asarray(scores_one, dtype=np.float32).reshape(-1)

        for link_index, (point_a, point_b) in enumerate(skeleton):
            if point_a >= len(keypoints_array) or point_b >= len(keypoints_array):
                continue
            if not (visible_array[point_a] and visible_array[point_b]):
                continue
            if scores_array[point_a] < kpt_thr or scores_array[point_b] < kpt_thr:
                continue

            x1, y1 = map(int, np.round(keypoints_array[point_a]))
            x2, y2 = map(int, np.round(keypoints_array[point_b]))
            if not (_in_bounds(x1, y1) and _in_bounds(x2, y2)):
                continue

            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                link_colors[link_index % len(link_colors)],
                thickness=max(1, thickness),
                lineType=cv2.LINE_AA,
            )

        for keypoint_index, (xy, is_visible, score_value) in enumerate(
            zip(keypoints_array, visible_array, scores_array)
        ):
            if not is_visible or score_value < kpt_thr:
                continue
            x, y = map(int, np.round(xy))
            if not _in_bounds(x, y):
                continue

            point_color = keypoint_colors[min(keypoint_index, len(keypoint_colors) - 1)]
            cv2.circle(
                image,
                (x, y),
                radius,
                point_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            if show_kpt_idx:
                cv2.putText(
                    image,
                    str(keypoint_index),
                    (x + radius, y - radius),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    point_color,
                    1,
                    cv2.LINE_AA,
                )

    return image
