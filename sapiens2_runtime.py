from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .pose_render_utils import visualize_keypoints

try:
    import folder_paths
except Exception:  # pragma: no cover - only unavailable outside ComfyUI
    folder_paths = None

TASK_CONFIGS = {
    "seg": {
        "family": "dense",
        "folder_name": "sapiens2_seg",
        "model_subdir": "seg",
        "checkpoint_suffix": "_seg.safetensors",
        "config_template": (
            "sapiens/dense/configs/seg/shutterstock_goliath/"
            "sapiens2_{size}_seg_shutterstock_goliath-1024x768.py"
        ),
    },
    "normal": {
        "family": "dense",
        "folder_name": "sapiens2_normal",
        "model_subdir": "normal",
        "checkpoint_suffix": "_normal.safetensors",
        "config_template": (
            "sapiens/dense/configs/normal/metasim_render_people/"
            "sapiens2_{size}_normal_metasim_render_people-1024x768.py"
        ),
    },
    "pointmap": {
        "family": "dense",
        "folder_name": "sapiens2_pointmap",
        "model_subdir": "pointmap",
        "checkpoint_suffix": "_pointmap.safetensors",
        "config_template": (
            "sapiens/dense/configs/pointmap/render_people/"
            "sapiens2_{size}_pointmap_render_people-1024x768.py"
        ),
    },
    "pose": {
        "family": "pose",
        "folder_name": "sapiens2_pose",
        "model_subdir": "pose",
        "checkpoint_suffix": "_pose.safetensors",
        "config_template": (
            "sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/"
            "sapiens2_{size}_keypoints308_shutterstock_goliath_3po-1024x768.py"
        ),
    },
}

DETECTOR_CONFIG = {
    "folder_name": "sapiens2_detector",
    "model_subdir": "detector",
    "default_filename": "rtmdet_m.pth",
    "config_rel": "sapiens/pose/tools/vis/rtmdet_m_640-8xb32_coco-person.py",
}

SEG_CLASSES = [
    "Background",
    "Apparel",
    "Eyeglass",
    "Face_Neck",
    "Hair",
    "Left_Foot",
    "Left_Hand",
    "Left_Lower_Arm",
    "Left_Lower_Leg",
    "Left_Shoe",
    "Left_Sock",
    "Left_Upper_Arm",
    "Left_Upper_Leg",
    "Lower_Clothing",
    "Right_Foot",
    "Right_Hand",
    "Right_Lower_Arm",
    "Right_Lower_Leg",
    "Right_Shoe",
    "Right_Sock",
    "Right_Upper_Arm",
    "Right_Upper_Leg",
    "Torso",
    "Upper_Clothing",
    "Lower_Lip",
    "Upper_Lip",
    "Lower_Teeth",
    "Upper_Teeth",
    "Tongue",
]


def _build_palette(num_classes: int) -> torch.Tensor:
    colors = []
    for index in range(num_classes):
        colors.append(
            [
                (37 * index) % 255,
                (91 * index) % 255,
                (173 * index) % 255,
            ]
        )
    return torch.tensor(colors, dtype=torch.float32) / 255.0


SEG_PALETTE = _build_palette(len(SEG_CLASSES))
MODEL_SIZE_CHOICES = ["auto", "0.4b", "0.8b", "1b", "5b"]
POINTMAP_PREVIEW_MODE_CHOICES = ["turbo_inverse_depth", "grayscale_z"]


@dataclass
class Sapiens2ModelBundle:
    task: str
    model_size: str
    checkpoint_name: str
    checkpoint_path: str
    config_path: str
    repo_root: str
    device: str
    model: Any


@dataclass
class Sapiens2DetectorBundle:
    checkpoint_name: str
    checkpoint_path: str
    config_path: str
    repo_root: str
    device: str
    detector: Any


def register_comfy_model_folders() -> None:
    if folder_paths is None:
        return

    try:
        checkpoints_dir = Path(folder_paths.get_folder_paths("checkpoints")[0])
    except Exception:
        return

    models_root = checkpoints_dir.parent
    sapiens_root = models_root / "sapiens2"
    for spec in TASK_CONFIGS.values():
        target = sapiens_root / spec["model_subdir"]
        target.mkdir(parents=True, exist_ok=True)
        folder_paths.add_model_folder_path(spec["folder_name"], str(target), is_default=True)

    detector_target = sapiens_root / DETECTOR_CONFIG["model_subdir"]
    detector_target.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path(
        DETECTOR_CONFIG["folder_name"],
        str(detector_target),
        is_default=True,
    )


register_comfy_model_folders()


def get_checkpoint_choices(task: str) -> list[str]:
    if folder_paths is None:
        return ["<run inside ComfyUI to list checkpoints>"]

    folder_name = TASK_CONFIGS[task]["folder_name"]
    try:
        names = folder_paths.get_filename_list(folder_name)
    except Exception:
        names = []
    return names or ["<no checkpoints found>"]


def get_detector_checkpoint_choices() -> list[str]:
    if folder_paths is None:
        return ["<run inside ComfyUI to list detector checkpoints>"]

    try:
        names = folder_paths.get_filename_list(DETECTOR_CONFIG["folder_name"])
    except Exception:
        names = []
    return names or ["<no detector checkpoint found>"]


def infer_model_size_from_name(filename: str) -> str:
    lowered = filename.lower()
    for candidate in ("0.4b", "0.8b", "1b", "5b"):
        if f"_{candidate}_" in lowered or f"2-{candidate}" in lowered:
            return candidate
    raise ValueError(
        f"Could not infer model size from checkpoint name '{filename}'. "
        "Pick an explicit model size instead of 'auto'."
    )


def _resolve_repo_root(repo_root_text: str) -> Path:
    candidates: list[Path] = []
    if repo_root_text.strip():
        candidates.append(Path(repo_root_text).expanduser())

    env_path = os.environ.get("SAPIENS2_REPO_ROOT", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            here / ".deps" / "sapiens2",
            here / "sapiens2",
            here.parent / "sapiens2",
        ]
    )

    for candidate in candidates:
        if (candidate / "sapiens").is_dir() and (candidate / "README.md").exists():
            return candidate.resolve()

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate the official facebookresearch/sapiens2 repo.\n"
        "Set `repo_root` in the node UI, set `SAPIENS2_REPO_ROOT`, or clone the repo "
        "to `.deps/sapiens2` inside this custom node.\n"
        f"Searched:\n{searched}"
    )


def _ensure_repo_on_path(repo_root: Path) -> None:
    root = str(repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_task_init_model(repo_root: Path, task: str):
    _ensure_repo_on_path(repo_root)
    family = TASK_CONFIGS[task]["family"]
    if family == "pose":
        return importlib.import_module("sapiens.pose.models").init_model
    return importlib.import_module("sapiens.dense.src.models.init_model").init_model


def _import_pose_helpers(repo_root: Path):
    _ensure_repo_on_path(repo_root)
    pose_datasets = importlib.import_module("sapiens.pose.datasets")
    pose_evaluators = importlib.import_module("sapiens.pose.evaluators")
    return pose_datasets.parse_pose_metainfo, pose_datasets.UDPHeatmap, pose_evaluators.nms


def _mmdet_install_message(original_error: Exception) -> str:
    return (
        "Sapiens2 pose detection requires MMDetection dependencies "
        "(`mmdet`, `mmengine`, and `mmcv` or `mmcv-lite`). "
        "In the ComfyUI Python environment, install them with a command such as:\n"
        "`pip install -U openmim && mim install mmengine && mim install mmcv-lite && mim install mmdet`\n"
        f"Original import error: {original_error}"
    )


def _import_mmdet_apis():
    sys.modules["mmpretrain"] = None
    try:
        mmdet_apis = importlib.import_module("mmdet.apis")
        mmdet_datasets = importlib.import_module("mmdet.datasets")
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise ImportError(_mmdet_install_message(exc)) from exc
    return mmdet_apis.init_detector, mmdet_apis.inference_detector, mmdet_datasets.transforms


def _resolve_checkpoint_path(task: str, checkpoint_name: str) -> Path:
    if checkpoint_name.startswith("<"):
        raise FileNotFoundError(
            f"No usable checkpoint selected for task '{task}'. "
            f"Place a checkpoint in ComfyUI/models/sapiens2/{TASK_CONFIGS[task]['model_subdir']}/ first."
        )

    checkpoint_path = Path(checkpoint_name).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path

    if folder_paths is None:
        raise FileNotFoundError(
            "Relative checkpoint names require ComfyUI folder_paths support."
        )

    folder_name = TASK_CONFIGS[task]["folder_name"]
    return Path(folder_paths.get_full_path_or_raise(folder_name, checkpoint_name))


def _resolve_detector_checkpoint_path(checkpoint_name: str) -> Path:
    if checkpoint_name.startswith("<"):
        raise FileNotFoundError(
            "No usable detector checkpoint selected. "
            "Place `rtmdet_m.pth` in ComfyUI/models/sapiens2/detector/ first."
        )

    checkpoint_path = Path(checkpoint_name).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path

    if folder_paths is None:
        raise FileNotFoundError(
            "Relative detector checkpoint names require ComfyUI folder_paths support."
        )

    return Path(
        folder_paths.get_full_path_or_raise(
            DETECTOR_CONFIG["folder_name"],
            checkpoint_name,
        )
    )


def _resolve_config_path(repo_root: Path, task: str, model_size: str) -> Path:
    spec = TASK_CONFIGS[task]
    config_rel = spec["config_template"].format(size=model_size)
    config_path = repo_root / config_rel
    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected official Sapiens2 config at: {config_path}"
        )
    return config_path


def _resolve_detector_config_path(repo_root: Path) -> Path:
    config_path = repo_root / DETECTOR_CONFIG["config_rel"]
    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected official RTMDet config at: {config_path}"
        )
    return config_path


def _pose_metainfo_path(repo_root: Path) -> Path:
    return repo_root / "sapiens" / "pose" / "configs" / "_base_" / "keypoints308.py"


def _prepare_pose_model(repo_root: Path, model: Any) -> Any:
    parse_pose_metainfo, UDPHeatmap, _ = _import_pose_helpers(repo_root)
    if int(getattr(model.cfg, "num_keypoints", 0)) != 308:
        raise ValueError(
            "This MVP currently supports the official 308-keypoint Sapiens2 pose models only."
        )

    metainfo_path = _pose_metainfo_path(repo_root)
    model.pose_metainfo = parse_pose_metainfo({"from_file": str(metainfo_path)})

    codec_config = dict(model.cfg.codec)
    codec_type = codec_config.pop("type")
    if codec_type != "UDPHeatmap":
        raise ValueError(
            f"Unsupported pose codec '{codec_type}'. Only UDPHeatmap is supported."
        )
    model.codec = UDPHeatmap(**codec_config)
    return model


def _patch_mmdet_pipeline(cfg: Any, transforms_module: Any):
    if "test_dataloader" not in cfg:
        return cfg

    pipeline = cfg.test_dataloader.dataset.pipeline
    available = dir(transforms_module)
    for trans in pipeline:
        if isinstance(trans, dict):
            trans_type = trans.get("type")
            if trans_type in available:
                trans["type"] = "mmdet." + trans_type
    return cfg


def load_model_bundle(
    *,
    repo_root_text: str,
    task: str,
    checkpoint_name: str,
    model_size: str,
    device: str,
) -> Sapiens2ModelBundle:
    repo_root = _resolve_repo_root(repo_root_text)
    checkpoint_path = _resolve_checkpoint_path(task, checkpoint_name)
    resolved_model_size = (
        infer_model_size_from_name(checkpoint_path.name)
        if model_size == "auto"
        else model_size
    )
    config_path = _resolve_config_path(repo_root, task, resolved_model_size)
    init_model = _import_task_init_model(repo_root, task)
    model = init_model(str(config_path), str(checkpoint_path), device=device)
    if task == "pose":
        model = _prepare_pose_model(repo_root, model)
    return Sapiens2ModelBundle(
        task=task,
        model_size=resolved_model_size,
        checkpoint_name=checkpoint_path.name,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        repo_root=str(repo_root),
        device=device,
        model=model,
    )


def load_detector_bundle(
    *,
    repo_root_text: str,
    checkpoint_name: str,
    device: str,
) -> Sapiens2DetectorBundle:
    repo_root = _resolve_repo_root(repo_root_text)
    checkpoint_path = _resolve_detector_checkpoint_path(checkpoint_name)
    config_path = _resolve_detector_config_path(repo_root)
    init_detector, _, transforms_module = _import_mmdet_apis()
    detector = init_detector(str(config_path), str(checkpoint_path), device=device)
    detector.cfg = _patch_mmdet_pipeline(detector.cfg, transforms_module)
    return Sapiens2DetectorBundle(
        checkpoint_name=checkpoint_path.name,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        repo_root=str(repo_root),
        device=device,
        detector=detector,
    )


def _comfy_image_to_bgr_uint8(image_rgb: torch.Tensor) -> np.ndarray:
    image_rgb = image_rgb.detach().cpu().clamp(0.0, 1.0)
    image_uint8 = (image_rgb.numpy() * 255.0).round().astype(np.uint8)
    return image_uint8[:, :, ::-1].copy()


def _image_to_uint8_rgb(image_rgb: torch.Tensor) -> np.ndarray:
    image_rgb = image_rgb.detach().cpu().clamp(0.0, 1.0)
    return (image_rgb.numpy() * 255.0).round().astype(np.uint8)


def _rgb_uint8_to_comfy_image(image_rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image_rgb)).to(dtype=torch.float32) / 255.0


def _run_pipeline(bundle: Sapiens2ModelBundle, image_bgr: np.ndarray) -> dict[str, Any]:
    data = bundle.model.pipeline(dict(img=image_bgr))
    return bundle.model.data_preprocessor(data)


def _padding_from_data(data: dict[str, Any]) -> tuple[int, int, int, int]:
    data_samples = data["data_samples"]
    padding = data_samples["meta"]["padding_size"]
    if isinstance(padding, torch.Tensor):
        padding = padding.tolist()
    return tuple(int(value) for value in padding)


def _broadcast_mask(mask: torch.Tensor | None, batch_size: int, height: int, width: int) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch_size, height, width), dtype=torch.float32)

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError("MASK inputs must be shaped [H,W] or [B,H,W].")
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1)
    if mask.shape[0] != batch_size:
        raise ValueError(
            f"MASK batch ({mask.shape[0]}) does not match image batch ({batch_size})."
        )
    if mask.shape[1] != height or mask.shape[2] != width:
        mask = F.interpolate(
            mask.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return (mask > 0.5).to(dtype=torch.float32, device="cpu")


def _normalize_boxes_tensor(
    boxes: torch.Tensor | np.ndarray | list,
    image_h: int,
    image_w: int,
) -> torch.Tensor:
    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).detach().cpu()
    if boxes_tensor.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    boxes_tensor = boxes_tensor.reshape(-1, 4)
    boxes_tensor[:, 0::2] = boxes_tensor[:, 0::2].clamp(0, max(image_w - 1, 0))
    boxes_tensor[:, 1::2] = boxes_tensor[:, 1::2].clamp(0, max(image_h - 1, 0))
    return boxes_tensor


def _normalize_scores_tensor(
    scores: torch.Tensor | np.ndarray | list | None,
    expected_len: int,
) -> torch.Tensor:
    if expected_len == 0:
        return torch.empty((0,), dtype=torch.float32)
    if scores is None:
        return torch.ones((expected_len,), dtype=torch.float32)

    scores_tensor = torch.as_tensor(scores, dtype=torch.float32).detach().cpu().reshape(-1)
    if scores_tensor.numel() != expected_len:
        raise ValueError(
            f"BBOX scores length ({scores_tensor.numel()}) does not match boxes length ({expected_len})."
        )
    return scores_tensor


def _full_image_box(image_h: int, image_w: int) -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, float(max(image_w - 1, 0)), float(max(image_h - 1, 0))]],
        dtype=torch.float32,
    )


def _coerce_metadata_sequence(
    value: Any,
    batch_size: int,
    field_name: str,
):
    if isinstance(value, (list, tuple)):
        sequence = list(value)
    elif batch_size == 1:
        sequence = [value]
    else:
        raise ValueError(f"{field_name} must provide one entry per image in the batch.")
    if len(sequence) != batch_size:
        raise ValueError(
            f"{field_name} length ({len(sequence)}) does not match image batch ({batch_size})."
        )
    return sequence


def _render_bbox_preview(
    image_rgb: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    preview = _image_to_uint8_rgb(image_rgb)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (64, 255, 64), 2, cv2.LINE_AA)
        if index < len(scores):
            label = f"{float(scores[index]):.2f}"
            label_y = max(y1 - 6, 12)
            cv2.putText(
                preview,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return _rgb_uint8_to_comfy_image(preview)


def overlay_segmentation(image: torch.Tensor, labels: torch.Tensor, alpha: float) -> torch.Tensor:
    palette = SEG_PALETTE.to(image.device)
    colors = palette[labels.long().clamp(0, palette.shape[0] - 1)]
    return (image * (1.0 - alpha) + colors * alpha).clamp(0.0, 1.0)


def _apply_background(
    original_image: torch.Tensor,
    rendered_image: torch.Tensor,
    mask: torch.Tensor,
    preserve_background: bool,
) -> torch.Tensor:
    if preserve_background:
        background = original_image
    else:
        background = torch.zeros_like(original_image)
    return torch.where(mask.unsqueeze(-1) > 0.5, rendered_image, background)


def run_segmentation(
    bundle: Sapiens2ModelBundle,
    image_batch: torch.Tensor,
    overlay_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    previews = []
    masks = []
    labels_batch = []

    for image_rgb in image_batch:
        image_bgr = _comfy_image_to_bgr_uint8(image_rgb)
        data = _run_pipeline(bundle, image_bgr)
        inputs = data["inputs"]
        with torch.inference_mode():
            logits = bundle.model(inputs)
        logits = F.interpolate(
            logits,
            size=image_bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        labels = logits.argmax(dim=1).squeeze(0).cpu()
        mask = (labels > 0).float()
        preview = overlay_segmentation(image_rgb, labels, overlay_alpha)
        previews.append(preview)
        masks.append(mask)
        labels_batch.append(labels)

    preview_batch = torch.stack(previews, dim=0)
    mask_batch = torch.stack(masks, dim=0)
    labels_tensor = torch.stack(labels_batch, dim=0)
    metadata = {
        "task": "seg",
        "labels": labels_tensor,
        "classes": SEG_CLASSES,
        "model_size": bundle.model_size,
        "checkpoint_name": bundle.checkpoint_name,
    }
    return preview_batch, mask_batch, metadata


def run_normal(
    bundle: Sapiens2ModelBundle,
    image_batch: torch.Tensor,
    mask: torch.Tensor | None,
    preserve_background: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    previews = []
    resolved_mask = _broadcast_mask(
        mask,
        batch_size=image_batch.shape[0],
        height=image_batch.shape[1],
        width=image_batch.shape[2],
    )

    for index, image_rgb in enumerate(image_batch):
        image_bgr = _comfy_image_to_bgr_uint8(image_rgb)
        data = _run_pipeline(bundle, image_bgr)
        inputs = data["inputs"]
        with torch.inference_mode():
            normal = bundle.model(inputs)
        normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(min=1e-8)
        pad_left, pad_right, pad_top, pad_bottom = _padding_from_data(data)
        normal = normal[
            :,
            :,
            pad_top : inputs.shape[2] - pad_bottom,
            pad_left : inputs.shape[3] - pad_right,
        ]
        normal = F.interpolate(
            normal,
            size=(image_bgr.shape[0], image_bgr.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        preview = ((normal.squeeze(0).cpu().movedim(0, -1) + 1.0) * 0.5).clamp(0.0, 1.0)
        preview = _apply_background(
            image_rgb,
            preview,
            resolved_mask[index],
            preserve_background=preserve_background,
        )
        previews.append(preview)

    return torch.stack(previews, dim=0), resolved_mask


def _normalize_depth_for_preview(depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    preview = torch.zeros_like(depth)
    valid = mask > 0.5
    if valid.any():
        values = depth[valid]
        min_value = torch.quantile(values, 0.01)
        max_value = torch.quantile(values, 0.99)
        if (max_value - min_value).abs() < 1e-6:
            min_value = values.min()
            max_value = values.max()
        denom = (max_value - min_value).clamp(min=1e-6)
        preview[valid] = 1.0 - ((depth[valid] - min_value) / denom)
    return preview.clamp(0.0, 1.0)


def _normalize_inverse_depth_for_preview(depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    preview = torch.zeros_like(depth)
    valid = mask > 0.5
    if valid.any():
        depth_values = depth[valid].abs().clamp(min=1e-6)
        inverse_depth = 1.0 / depth_values
        min_inverse_depth = torch.quantile(inverse_depth, 0.01)
        max_inverse_depth = torch.quantile(inverse_depth, 0.99)
        if (max_inverse_depth - min_inverse_depth).abs() < 1e-6:
            min_inverse_depth = inverse_depth.min()
            max_inverse_depth = inverse_depth.max()
        denom = (max_inverse_depth - min_inverse_depth).clamp(min=1e-6)
        preview[valid] = (inverse_depth - min_inverse_depth) / denom
    return preview.clamp(0.0, 1.0)


def _colorize_with_turbo(preview: torch.Tensor) -> torch.Tensor:
    preview_uint8 = (preview.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    preview_bgr = cv2.applyColorMap(preview_uint8, cv2.COLORMAP_TURBO)
    preview_rgb = preview_bgr[:, :, ::-1].copy()
    return torch.from_numpy(preview_rgb).to(dtype=torch.float32) / 255.0


def _render_pointmap_preview(
    depth: torch.Tensor,
    mask: torch.Tensor,
    preview_mode: str,
) -> torch.Tensor:
    if preview_mode == "grayscale_z":
        preview = _normalize_depth_for_preview(depth, mask)
        return preview.unsqueeze(-1).repeat(1, 1, 3)
    if preview_mode == "turbo_inverse_depth":
        preview = _normalize_inverse_depth_for_preview(depth, mask)
        return _colorize_with_turbo(preview)
    raise ValueError(
        f"Unknown pointmap preview mode '{preview_mode}'. "
        f"Expected one of: {', '.join(POINTMAP_PREVIEW_MODE_CHOICES)}"
    )


def run_pointmap(
    bundle: Sapiens2ModelBundle,
    image_batch: torch.Tensor,
    mask: torch.Tensor | None,
    preserve_background: bool,
    preview_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    previews = []
    pointmaps = []
    resolved_mask = _broadcast_mask(
        mask,
        batch_size=image_batch.shape[0],
        height=image_batch.shape[1],
        width=image_batch.shape[2],
    )

    for index, image_rgb in enumerate(image_batch):
        image_bgr = _comfy_image_to_bgr_uint8(image_rgb)
        data = _run_pipeline(bundle, image_bgr)
        inputs = data["inputs"]
        with torch.inference_mode():
            pointmap, scale = bundle.model(inputs)
        pointmap = pointmap / scale
        pad_left, pad_right, pad_top, pad_bottom = _padding_from_data(data)
        pointmap = pointmap[
            :,
            :,
            pad_top : inputs.shape[2] - pad_bottom,
            pad_left : inputs.shape[3] - pad_right,
        ]
        pointmap = F.interpolate(
            pointmap,
            size=(image_bgr.shape[0], image_bgr.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        pointmap_cpu = pointmap.squeeze(0).cpu()
        depth = pointmap_cpu[2]
        preview = _render_pointmap_preview(depth, resolved_mask[index], preview_mode)
        preview = _apply_background(
            image_rgb,
            preview,
            resolved_mask[index],
            preserve_background=preserve_background,
        )
        previews.append(preview)
        pointmaps.append(pointmap_cpu)

    pointmap_tensor = torch.stack(pointmaps, dim=0)
    metadata = {
        "task": "pointmap",
        "pointmap": pointmap_tensor,
        "mask": resolved_mask,
        "model_size": bundle.model_size,
        "checkpoint_name": bundle.checkpoint_name,
        "preview_mode": preview_mode,
    }
    return torch.stack(previews, dim=0), resolved_mask, metadata


def _detect_person_boxes_for_image(
    bundle: Sapiens2DetectorBundle,
    image_bgr: np.ndarray,
    bbox_thr: float,
    nms_thr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, inference_detector, _ = _import_mmdet_apis()
    _, _, nms = _import_pose_helpers(Path(bundle.repo_root))

    det_result = inference_detector(bundle.detector, image_bgr)
    pred_instance = det_result.pred_instances.cpu().numpy()
    if pred_instance.bboxes.shape[0] == 0:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)

    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]),
        axis=1,
    )
    keep = np.logical_and(pred_instance.labels == 0, pred_instance.scores > bbox_thr)
    bboxes = bboxes[keep]
    if len(bboxes) == 0:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)

    kept_indices = nms(bboxes, nms_thr)
    bboxes = bboxes[kept_indices]
    boxes_tensor = torch.from_numpy(np.ascontiguousarray(bboxes[:, :4])).to(dtype=torch.float32)
    scores_tensor = torch.from_numpy(np.ascontiguousarray(bboxes[:, 4])).to(dtype=torch.float32)
    return boxes_tensor, scores_tensor


def run_person_detection(
    bundle: Sapiens2DetectorBundle,
    image_batch: torch.Tensor,
    bbox_thr: float,
    nms_thr: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    previews = []
    boxes_batch = []
    scores_batch = []
    image_sizes = []

    for image_rgb in image_batch:
        image_bgr = _comfy_image_to_bgr_uint8(image_rgb)
        image_h, image_w = image_bgr.shape[:2]
        boxes, scores = _detect_person_boxes_for_image(bundle, image_bgr, bbox_thr, nms_thr)
        boxes = _normalize_boxes_tensor(boxes, image_h, image_w)
        scores = _normalize_scores_tensor(scores, boxes.shape[0])
        previews.append(_render_bbox_preview(image_rgb, boxes, scores))
        boxes_batch.append(boxes)
        scores_batch.append(scores)
        image_sizes.append((image_h, image_w))

    metadata = {
        "task": "bboxes",
        "boxes": boxes_batch,
        "scores": scores_batch,
        "image_sizes": image_sizes,
        "source": "detector",
    }
    return torch.stack(previews, dim=0), metadata


def _coerce_bboxes_metadata(
    bboxes: dict[str, Any],
    image_batch: torch.Tensor,
) -> dict[str, Any]:
    if bboxes.get("task") != "bboxes":
        raise ValueError("Sapiens2 pose nodes expect a SAPIENS2_BBOXES input.")

    batch_size = image_batch.shape[0]
    boxes_values = _coerce_metadata_sequence(bboxes.get("boxes"), batch_size, "boxes")
    scores_values = _coerce_metadata_sequence(
        bboxes.get("scores", [None] * batch_size),
        batch_size,
        "scores",
    )

    image_h = int(image_batch.shape[1])
    image_w = int(image_batch.shape[2])
    normalized_boxes = []
    normalized_scores = []
    image_sizes = []
    for boxes_value, scores_value in zip(boxes_values, scores_values):
        boxes_tensor = _normalize_boxes_tensor(boxes_value, image_h, image_w)
        scores_tensor = _normalize_scores_tensor(scores_value, boxes_tensor.shape[0])
        normalized_boxes.append(boxes_tensor)
        normalized_scores.append(scores_tensor)
        image_sizes.append((image_h, image_w))

    return {
        "task": "bboxes",
        "boxes": normalized_boxes,
        "scores": normalized_scores,
        "image_sizes": image_sizes,
        "source": str(bboxes.get("source", "external")),
    }


def _resolve_pose_bboxes(
    image_batch: torch.Tensor,
    detector_bundle: Sapiens2DetectorBundle | None,
    bboxes: dict[str, Any] | None,
    bbox_thr: float,
    nms_thr: float,
    fallback_full_image_bbox: bool,
) -> dict[str, Any]:
    batch_size = image_batch.shape[0]
    image_h = int(image_batch.shape[1])
    image_w = int(image_batch.shape[2])

    if bboxes is not None:
        resolved = _coerce_bboxes_metadata(bboxes, image_batch)
    elif detector_bundle is not None:
        _, resolved = run_person_detection(detector_bundle, image_batch, bbox_thr, nms_thr)
    elif fallback_full_image_bbox:
        full_box = _full_image_box(image_h, image_w)
        resolved = {
            "task": "bboxes",
            "boxes": [full_box.clone() for _ in range(batch_size)],
            "scores": [torch.ones((1,), dtype=torch.float32) for _ in range(batch_size)],
            "image_sizes": [(image_h, image_w) for _ in range(batch_size)],
            "source": "full_image_fallback",
        }
    else:
        raise ValueError(
            "Sapiens2 Pose Estimation needs either a detector input, a SAPIENS2_BBOXES input, "
            "or `fallback_full_image_bbox=True`."
        )

    if fallback_full_image_bbox:
        for index, boxes in enumerate(resolved["boxes"]):
            if boxes.shape[0] == 0:
                resolved["boxes"][index] = _full_image_box(image_h, image_w)
                resolved["scores"][index] = torch.ones((1,), dtype=torch.float32)
    return resolved


def _run_pose_for_boxes(
    bundle: Sapiens2ModelBundle,
    image_bgr: np.ndarray,
    boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.shape[0] == 0:
        num_keypoints = int(bundle.model.pose_metainfo["num_keypoints"])
        return (
            torch.empty((0, num_keypoints, 2), dtype=torch.float32),
            torch.empty((0, num_keypoints), dtype=torch.float32),
        )

    inputs_list = []
    data_samples_list = []
    for box in boxes:
        data_info = dict(img=image_bgr)
        data_info["bbox"] = box.detach().cpu().numpy()[None].astype(np.float32)
        data_info["bbox_score"] = np.ones(1, dtype=np.float32)
        data = bundle.model.pipeline(data_info)
        data = bundle.model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        data_samples_list.append(data["data_samples"])

    inputs = torch.cat(inputs_list, dim=0)
    with torch.inference_mode():
        predictions = bundle.model(inputs)
        if bundle.model.cfg.val_cfg is not None and bundle.model.cfg.val_cfg.get("flip_test", False):
            flipped = bundle.model(inputs.flip(-1))
            flipped = flipped.flip(-1)
            flip_indices = bundle.model.pose_metainfo["flip_indices"]
            if len(flip_indices) != flipped.shape[1]:
                raise ValueError("Pose flip-test metadata does not match model output channels.")
            flipped = flipped[:, flip_indices]
            predictions = (predictions + flipped) / 2.0

    predictions = predictions.detach().cpu().numpy()
    keypoints = []
    keypoint_scores = []
    for index, data_samples in enumerate(data_samples_list):
        keypoints_i, keypoint_scores_i = bundle.model.codec.decode(predictions[index])
        input_size = np.asarray(data_samples["meta"]["input_size"], dtype=np.float32)
        bbox_center = np.asarray(data_samples["meta"]["bbox_center"], dtype=np.float32)
        bbox_scale = np.asarray(data_samples["meta"]["bbox_scale"], dtype=np.float32)
        keypoints_i = (
            keypoints_i / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        )
        keypoints.append(torch.from_numpy(np.asarray(keypoints_i[0], dtype=np.float32)))
        keypoint_scores.append(
            torch.from_numpy(np.asarray(keypoint_scores_i[0], dtype=np.float32))
        )

    return torch.stack(keypoints, dim=0), torch.stack(keypoint_scores, dim=0)


def _keypoint_names_from_metainfo(pose_metainfo: dict[str, Any]) -> list[str]:
    id_to_name = pose_metainfo["keypoint_id2name"]
    return [id_to_name[index] for index in sorted(id_to_name.keys())]


def _render_pose_overlay(
    image_rgb: torch.Tensor,
    keypoints: torch.Tensor,
    keypoint_scores: torch.Tensor,
    pose_metainfo: dict[str, Any],
    kpt_thr: float,
    radius: int,
    thickness: int,
) -> torch.Tensor:
    if keypoints.shape[0] == 0:
        return image_rgb.detach().cpu().clamp(0.0, 1.0)

    image_uint8 = _image_to_uint8_rgb(image_rgb)
    overlay_uint8 = visualize_keypoints(
        image=image_uint8,
        keypoints=[tensor.numpy() for tensor in keypoints],
        keypoints_visible=[np.ones((keypoint_scores.shape[1],), dtype=bool) for _ in range(keypoint_scores.shape[0])],
        keypoint_scores=[tensor.numpy() for tensor in keypoint_scores],
        radius=radius,
        thickness=thickness,
        kpt_thr=kpt_thr,
        skeleton=pose_metainfo["skeleton_links"],
        kpt_color=pose_metainfo["keypoint_colors"],
        link_color=pose_metainfo["skeleton_link_colors"],
    )
    return _rgb_uint8_to_comfy_image(overlay_uint8)


def run_pose(
    bundle: Sapiens2ModelBundle,
    image_batch: torch.Tensor,
    detector_bundle: Sapiens2DetectorBundle | None,
    bboxes: dict[str, Any] | None,
    bbox_thr: float,
    nms_thr: float,
    fallback_full_image_bbox: bool,
    kpt_thr: float,
    radius: int,
    thickness: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if bundle.task != "pose":
        raise ValueError("run_pose requires a Sapiens2 pose model bundle.")

    resolved_bboxes = _resolve_pose_bboxes(
        image_batch=image_batch,
        detector_bundle=detector_bundle,
        bboxes=bboxes,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        fallback_full_image_bbox=fallback_full_image_bbox,
    )

    pose_metainfo = bundle.model.pose_metainfo
    keypoint_names = _keypoint_names_from_metainfo(pose_metainfo)

    overlays = []
    instances = []
    for image_rgb, boxes in zip(image_batch, resolved_bboxes["boxes"]):
        image_bgr = _comfy_image_to_bgr_uint8(image_rgb)
        boxes_tensor = _normalize_boxes_tensor(boxes, image_bgr.shape[0], image_bgr.shape[1])
        keypoints_tensor, keypoint_scores_tensor = _run_pose_for_boxes(bundle, image_bgr, boxes_tensor)
        overlays.append(
            _render_pose_overlay(
                image_rgb,
                keypoints_tensor,
                keypoint_scores_tensor,
                pose_metainfo,
                kpt_thr=kpt_thr,
                radius=radius,
                thickness=thickness,
            )
        )
        instances.append(
            {
                "bboxes": boxes_tensor,
                "keypoints": keypoints_tensor,
                "keypoint_scores": keypoint_scores_tensor,
            }
        )

    metadata = {
        "task": "pose",
        "instances": instances,
        "num_keypoints": len(keypoint_names),
        "keypoint_names": keypoint_names,
        "skeleton_links": list(pose_metainfo["skeleton_links"]),
        "model_size": bundle.model_size,
        "checkpoint_name": bundle.checkpoint_name,
        "source": resolved_bboxes["source"],
    }
    return torch.stack(overlays, dim=0), metadata


def parse_part_list(part_list: str) -> list[int]:
    if not part_list.strip():
        raise ValueError("part_list cannot be empty.")

    resolved = []
    name_map = {name.lower(): index for index, name in enumerate(SEG_CLASSES)}
    for raw_token in part_list.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token.isdigit():
            value = int(token)
        else:
            key = token.lower()
            if key not in name_map:
                raise ValueError(
                    f"Unknown part '{token}'. Use ids like '22,23' or names like 'Torso,Upper_Clothing'."
                )
            value = name_map[key]
        if value < 0 or value >= len(SEG_CLASSES):
            raise ValueError(f"Part id {value} is out of range 0-{len(SEG_CLASSES) - 1}.")
        resolved.append(value)

    if not resolved:
        raise ValueError("part_list did not resolve to any valid part ids.")
    return sorted(set(resolved))
