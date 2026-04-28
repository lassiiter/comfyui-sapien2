from __future__ import annotations

from typing import Any

import torch

from .sapiens2_runtime import (
    MODEL_SIZE_CHOICES,
    POINTMAP_PREVIEW_MODE_CHOICES,
    Sapiens2DetectorBundle,
    Sapiens2ModelBundle,
    get_checkpoint_choices,
    get_detector_checkpoint_choices,
    load_detector_bundle,
    load_model_bundle,
    parse_part_list,
    run_normal,
    run_person_detection,
    run_pointmap,
    run_pose,
    run_segmentation,
    save_pointmap_ply,
    save_pose_json,
)

_MODEL_CACHE: dict[tuple[str, str, str, str, str], Sapiens2ModelBundle] = {}
_DETECTOR_CACHE: dict[tuple[str, str, str], Sapiens2DetectorBundle] = {}


class _BaseLoader:
    TASK = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_root": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "checkpoint_name": (get_checkpoint_choices(cls.TASK),),
                "model_size": (MODEL_SIZE_CHOICES,),
                "device": ("STRING", {"default": "cuda:0"}),
            }
        }

    RETURN_TYPES = ("SAPIENS2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Sapiens2"

    def load_model(
        self,
        repo_root: str,
        checkpoint_name: str,
        model_size: str,
        device: str,
    ):
        cache_key = (self.TASK, repo_root, checkpoint_name, model_size, device)
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = load_model_bundle(
                repo_root_text=repo_root,
                task=self.TASK,
                checkpoint_name=checkpoint_name,
                model_size=model_size,
                device=device,
            )
        return (_MODEL_CACHE[cache_key],)


class _BaseDetectorLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_root": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "checkpoint_name": (get_detector_checkpoint_choices(),),
                "device": ("STRING", {"default": "cuda:0"}),
            }
        }

    RETURN_TYPES = ("SAPIENS2_DETECTOR",)
    RETURN_NAMES = ("detector",)
    FUNCTION = "load_detector"
    CATEGORY = "Sapiens2"

    def load_detector(
        self,
        repo_root: str,
        checkpoint_name: str,
        device: str,
    ):
        cache_key = (repo_root, checkpoint_name, device)
        if cache_key not in _DETECTOR_CACHE:
            _DETECTOR_CACHE[cache_key] = load_detector_bundle(
                repo_root_text=repo_root,
                checkpoint_name=checkpoint_name,
                device=device,
            )
        return (_DETECTOR_CACHE[cache_key],)


class Sapiens2LoadSegModel(_BaseLoader):
    TASK = "seg"


class Sapiens2LoadNormalModel(_BaseLoader):
    TASK = "normal"


class Sapiens2LoadPointmapModel(_BaseLoader):
    TASK = "pointmap"


class Sapiens2LoadPoseModel(_BaseLoader):
    TASK = "pose"


class Sapiens2LoadPoseDetector(_BaseDetectorLoader):
    pass


class Sapiens2Segmentation:
    CATEGORY = "Sapiens2"
    FUNCTION = "segment"
    RETURN_TYPES = ("IMAGE", "MASK", "SAPIENS2_SEG_LABELS")
    RETURN_NAMES = ("overlay", "foreground_mask", "labels")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "overlay_alpha": (
                    "FLOAT",
                    {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    def segment(self, model: Sapiens2ModelBundle, image: torch.Tensor, overlay_alpha: float):
        if model.task != "seg":
            raise ValueError("Sapiens2Segmentation requires a segmentation model.")
        return run_segmentation(model, image, overlay_alpha)


class Sapiens2SegPartMask:
    CATEGORY = "Sapiens2"
    FUNCTION = "part_mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "labels": ("SAPIENS2_SEG_LABELS",),
                "part_list": (
                    "STRING",
                    {
                        "default": "22",
                        "multiline": False,
                    },
                ),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    def part_mask(self, labels: dict[str, Any], part_list: str, invert: bool):
        label_tensor = labels["labels"]
        part_ids = parse_part_list(part_list)
        mask = torch.zeros_like(label_tensor, dtype=torch.bool)
        for part_id in part_ids:
            mask |= label_tensor == part_id
        if invert:
            mask = ~mask
        return (mask.float(),)


class Sapiens2NormalEstimation:
    CATEGORY = "Sapiens2"
    FUNCTION = "estimate"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("normal_map", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "preserve_background": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    def estimate(
        self,
        model: Sapiens2ModelBundle,
        image: torch.Tensor,
        preserve_background: bool,
        mask: torch.Tensor | None = None,
    ):
        if model.task != "normal":
            raise ValueError("Sapiens2NormalEstimation requires a normal model.")
        return run_normal(model, image, mask, preserve_background)


class Sapiens2PointmapEstimation:
    CATEGORY = "Sapiens2"
    FUNCTION = "estimate"
    RETURN_TYPES = ("IMAGE", "MASK", "SAPIENS2_POINTMAP")
    RETURN_NAMES = ("depth_preview", "mask", "pointmap")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "preview_mode": (POINTMAP_PREVIEW_MODE_CHOICES,),
                "preserve_background": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    def estimate(
        self,
        model: Sapiens2ModelBundle,
        image: torch.Tensor,
        preview_mode: str,
        preserve_background: bool,
        mask: torch.Tensor | None = None,
    ):
        if model.task != "pointmap":
            raise ValueError("Sapiens2PointmapEstimation requires a pointmap model.")
        return run_pointmap(model, image, mask, preserve_background, preview_mode)


class Sapiens2PointmapDepthOnly:
    CATEGORY = "Sapiens2"
    FUNCTION = "depth_only"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("depth_z",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pointmap": ("SAPIENS2_POINTMAP",)}}

    def depth_only(self, pointmap: dict[str, Any]):
        if pointmap.get("task") != "pointmap":
            raise ValueError("Sapiens2PointmapDepthOnly expects a SAPIENS2_POINTMAP input.")
        return (pointmap["pointmap"][:, 2, :, :],)


class Sapiens2SavePointmapPLY:
    CATEGORY = "Sapiens2"
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_files",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pointmap": ("SAPIENS2_POINTMAP",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "sapiens2/pointmap",
                        "multiline": False,
                    },
                ),
                "use_mask": ("BOOLEAN", {"default": True}),
                "include_color": ("BOOLEAN", {"default": True}),
                "binary": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    def save(
        self,
        pointmap: dict[str, Any],
        filename_prefix: str,
        use_mask: bool,
        include_color: bool,
        binary: bool,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        saved_paths = save_pointmap_ply(
            pointmap=pointmap,
            filename_prefix=filename_prefix,
            image_batch=image,
            mask=mask,
            include_color=include_color,
            use_mask=use_mask,
            binary=binary,
        )
        return ("\n".join(saved_paths),)


class Sapiens2PersonDetection:
    CATEGORY = "Sapiens2"
    FUNCTION = "detect"
    RETURN_TYPES = ("IMAGE", "SAPIENS2_BBOXES")
    RETURN_NAMES = ("bbox_preview", "bboxes")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detector": ("SAPIENS2_DETECTOR",),
                "image": ("IMAGE",),
                "bbox_thr": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "nms_thr": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    def detect(
        self,
        detector: Sapiens2DetectorBundle,
        image: torch.Tensor,
        bbox_thr: float,
        nms_thr: float,
    ):
        return run_person_detection(detector, image, bbox_thr, nms_thr)


class Sapiens2PoseEstimation:
    CATEGORY = "Sapiens2"
    FUNCTION = "estimate"
    RETURN_TYPES = ("IMAGE", "SAPIENS2_POSE")
    RETURN_NAMES = ("overlay", "pose")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "bbox_thr": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "nms_thr": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "fallback_full_image_bbox": ("BOOLEAN", {"default": True}),
                "kpt_thr": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "radius": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1}),
            },
            "optional": {
                "detector": ("SAPIENS2_DETECTOR",),
                "bboxes": ("SAPIENS2_BBOXES",),
            },
        }

    def estimate(
        self,
        model: Sapiens2ModelBundle,
        image: torch.Tensor,
        bbox_thr: float,
        nms_thr: float,
        fallback_full_image_bbox: bool,
        kpt_thr: float,
        radius: int,
        thickness: int,
        detector: Sapiens2DetectorBundle | None = None,
        bboxes: dict[str, Any] | None = None,
    ):
        if model.task != "pose":
            raise ValueError("Sapiens2PoseEstimation requires a pose model.")
        return run_pose(
            model,
            image,
            detector,
            bboxes,
            bbox_thr,
            nms_thr,
            fallback_full_image_bbox,
            kpt_thr,
            radius,
            thickness,
        )


class Sapiens2SavePoseJSON:
    CATEGORY = "Sapiens2"
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_files",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("SAPIENS2_POSE",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "sapiens2/pose",
                        "multiline": False,
                    },
                ),
                "pretty_json": ("BOOLEAN", {"default": True}),
            }
        }

    def save(
        self,
        pose: dict[str, Any],
        filename_prefix: str,
        pretty_json: bool,
    ):
        saved_paths = save_pose_json(
            pose=pose,
            filename_prefix=filename_prefix,
            pretty_json=pretty_json,
        )
        return ("\n".join(saved_paths),)


NODE_CLASS_MAPPINGS = {
    "Sapiens2LoadSegModel": Sapiens2LoadSegModel,
    "Sapiens2LoadNormalModel": Sapiens2LoadNormalModel,
    "Sapiens2LoadPointmapModel": Sapiens2LoadPointmapModel,
    "Sapiens2LoadPoseModel": Sapiens2LoadPoseModel,
    "Sapiens2LoadPoseDetector": Sapiens2LoadPoseDetector,
    "Sapiens2Segmentation": Sapiens2Segmentation,
    "Sapiens2SegPartMask": Sapiens2SegPartMask,
    "Sapiens2NormalEstimation": Sapiens2NormalEstimation,
    "Sapiens2PointmapEstimation": Sapiens2PointmapEstimation,
    "Sapiens2PointmapDepthOnly": Sapiens2PointmapDepthOnly,
    "Sapiens2SavePointmapPLY": Sapiens2SavePointmapPLY,
    "Sapiens2PersonDetection": Sapiens2PersonDetection,
    "Sapiens2PoseEstimation": Sapiens2PoseEstimation,
    "Sapiens2SavePoseJSON": Sapiens2SavePoseJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sapiens2LoadSegModel": "Sapiens2 Load Seg Model",
    "Sapiens2LoadNormalModel": "Sapiens2 Load Normal Model",
    "Sapiens2LoadPointmapModel": "Sapiens2 Load Pointmap Model",
    "Sapiens2LoadPoseModel": "Sapiens2 Load Pose Model",
    "Sapiens2LoadPoseDetector": "Sapiens2 Load Pose Detector",
    "Sapiens2Segmentation": "Sapiens2 Segmentation",
    "Sapiens2SegPartMask": "Sapiens2 Seg Part Mask",
    "Sapiens2NormalEstimation": "Sapiens2 Normal Estimation",
    "Sapiens2PointmapEstimation": "Sapiens2 Pointmap Estimation",
    "Sapiens2PointmapDepthOnly": "Sapiens2 Pointmap Depth Only",
    "Sapiens2SavePointmapPLY": "Sapiens2 Save Pointmap PLY",
    "Sapiens2PersonDetection": "Sapiens2 Person Detection",
    "Sapiens2PoseEstimation": "Sapiens2 Pose Estimation",
    "Sapiens2SavePoseJSON": "Sapiens2 Save Pose JSON",
}
