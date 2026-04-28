# ComfyUI Sapiens2

ComfyUI custom nodes for Meta's `Sapiens2` models.

This project wraps the official `facebookresearch/sapiens2` codebase so you can run the current dense human-perception tasks from inside ComfyUI:

- segmentation
- surface normals
- pointmaps
- 308-keypoint pose estimation

It is a thin integration layer, not a reimplementation. The nodes use Meta's own configs, preprocessing, and model code, while exposing Comfy-friendly loaders, previews, masks, and structured outputs.

<img width="1470" height="523" alt="comfy" src="https://github.com/user-attachments/assets/c719eaba-1921-49fb-90f3-23854247cf21" />


## What This Repo Provides

- task-specific loader nodes with Comfy checkpoint dropdowns
- automatic config resolution based on task and model size
- segmentation overlays and part-mask extraction
- normal-map inference with optional masking
- pointmap inference with grayscale or turbo inverse-depth previews
- RTMDet-based person detection for pose
- top-down 308-keypoint pose estimation with rendered overlays
- structured intermediate objects so nodes can be chained together without rerunning every stage

## Node Summary

All nodes appear under the `Sapiens2` category in ComfyUI.

### Loader Nodes

- `Sapiens2 Load Seg Model`
- `Sapiens2 Load Normal Model`
- `Sapiens2 Load Pointmap Model`
- `Sapiens2 Load Pose Model`
- `Sapiens2 Load Pose Detector`

These nodes:

- look for checkpoints in `ComfyUI/models/sapiens2/<task>/`
- infer or accept the Sapiens2 model size
- resolve the matching official config file
- load and cache the model or detector bundle

### Dense Inference Nodes

- `Sapiens2 Segmentation`
  - outputs `overlay IMAGE`
  - outputs `foreground_mask MASK`
  - outputs `labels SAPIENS2_SEG_LABELS`

- `Sapiens2 Seg Part Mask`
  - converts segmentation labels into a Comfy `MASK`
  - accepts numeric ids like `22,23`
  - accepts class names like `Torso,Upper_Clothing`

- `Sapiens2 Normal Estimation`
  - outputs `normal_map IMAGE`
  - accepts an optional `mask`
  - supports `preserve_background`

- `Sapiens2 Pointmap Estimation`
  - outputs `depth_preview IMAGE`
  - outputs `mask MASK`
  - outputs `pointmap SAPIENS2_POINTMAP`
  - supports `preview_mode = turbo_inverse_depth` or `grayscale_z`
  - supports `preserve_background`

- `Sapiens2 Pointmap Depth Only`
  - extracts the raw pointmap `Z` plane as a `MASK`

### Pose Nodes

- `Sapiens2 Person Detection`
  - runs RTMDet person detection
  - outputs `bbox_preview IMAGE`
  - outputs `bboxes SAPIENS2_BBOXES`

- `Sapiens2 Pose Estimation`
  - runs top-down 308-keypoint pose inference
  - accepts either `detector` or precomputed `bboxes`
  - outputs `overlay IMAGE`
  - outputs `pose SAPIENS2_POSE`

## Important Dependency Model

This repo depends on a local clone of the official `facebookresearch/sapiens2` repository. It does not vendor Meta's full implementation.

The node will try to find the official repo in this order:

1. `repo_root` set in the loader UI
2. `SAPIENS2_REPO_ROOT` environment variable
3. `.deps/sapiens2` inside this custom node repo
4. `sapiens2` as a sibling directory

The easiest setup is:

```text
ComfyUI/custom_nodes/comfyui-sapien2/.deps/sapiens2
```

## Installation

### 1. Install This Repo As a Custom Node

Place this folder in:

```text
ComfyUI/custom_nodes/comfyui-sapien2
```

### 2. Clone Meta's Official Repo

From inside the custom node folder:

```powershell
git clone https://github.com/facebookresearch/sapiens2.git .deps\sapiens2
```

### 3. Install the Official Sapiens2 Package in Comfy's Python Environment

```powershell
cd C:\path\to\ComfyUI\custom_nodes\comfyui-sapien2\.deps\sapiens2
pip install -e .
```

Optional, if you also want Meta's extra pointmap tooling:

```powershell
pip install -e .[pointmap]
```

### 4. Install Pose Detector Dependencies

Pose support needs MMDetection-side dependencies in addition to the official Sapiens2 package.

A working install command for the current Comfy environment used during development was:

```powershell
pip install mmengine "mmcv-lite>=2.0.0rc4,<2.2.0" mmdet
```

That resolved to this working combination on the test machine:

- `torch 2.9.1+cu130`
- `mmengine 0.10.7`
- `mmcv 2.1.0`
- `mmdet 3.3.0`

If detector imports fail, start by checking `mmcv` compatibility first.

### 5. Put Checkpoints in Comfy's Model Folders

Create these folders if they do not exist:

```text
ComfyUI/models/sapiens2/seg
ComfyUI/models/sapiens2/normal
ComfyUI/models/sapiens2/pointmap
ComfyUI/models/sapiens2/pose
ComfyUI/models/sapiens2/detector
```

Expected checkpoint naming:

- `models/sapiens2/seg/sapiens2_<size>_seg.safetensors`
- `models/sapiens2/normal/sapiens2_<size>_normal.safetensors`
- `models/sapiens2/pointmap/sapiens2_<size>_pointmap.safetensors`
- `models/sapiens2/pose/sapiens2_<size>_pose.safetensors`
- `models/sapiens2/detector/rtmdet_m.pth`

### 6. Restart ComfyUI

Restart after:

- adding or replacing checkpoints
- changing custom-node Python files
- changing node inputs or outputs

## Official Configs Used

These are resolved automatically by the loaders.

### Segmentation

- `sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_<size>_seg_shutterstock_goliath-1024x768.py`

### Surface Normals

- `sapiens/dense/configs/normal/metasim_render_people/sapiens2_<size>_normal_metasim_render_people-1024x768.py`

### Pointmaps

- `sapiens/dense/configs/pointmap/render_people/sapiens2_<size>_pointmap_render_people-1024x768.py`

### Pose

- `sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_<size>_keypoints308_shutterstock_goliath_3po-1024x768.py`

### Pose Detector

- `sapiens/pose/tools/vis/rtmdet_m_640-8xb32_coco-person.py`

## Quick Test Workflows

### Segmentation

```text
Load Image
-> Sapiens2 Load Seg Model
-> Sapiens2 Segmentation
-> Preview Image
```

For part masking:

```text
Sapiens2 Segmentation.labels
-> Sapiens2 Seg Part Mask
-> Mask To Image
-> Preview Image
```

### Surface Normals

```text
Load Image
-> Sapiens2 Load Normal Model
-> Sapiens2 Normal Estimation
-> Preview Image
```

Optional:

- connect a segmentation-derived mask into `mask`
- set `preserve_background = true` to keep the original image outside the mask

### Pointmaps

```text
Load Image
-> Sapiens2 Load Pointmap Model
-> Sapiens2 Pointmap Estimation
-> Preview Image
```

Recommended settings:

- `preview_mode = turbo_inverse_depth`
- use a person/body mask for cleaner normalization
- use `preserve_background = true` if you want the original image outside the mask

Notes:

- `depth_preview` is only a visualization of the pointmap, not the raw 3D output
- `turbo_inverse_depth` is the paper-style view
- `grayscale_z` is a simpler depth-style preview

### Pose

Minimal path:

```text
Load Image
-> Sapiens2 Load Pose Model
-> Sapiens2 Load Pose Detector
-> Sapiens2 Pose Estimation
-> Preview Image
```

Reusable detection path:

```text
Load Image
-> Sapiens2 Load Pose Detector
-> Sapiens2 Person Detection
-> Sapiens2 Pose Estimation
-> Preview Image
```

When `bboxes` is connected to `Sapiens2 Pose Estimation`, it takes precedence over the optional `detector` input.

## Official References

- [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2)
- [Sapiens2 model index](https://huggingface.co/facebook/sapiens2)
- [Seg docs](https://github.com/facebookresearch/sapiens2/blob/main/docs/SEG.md)
- [Normal docs](https://github.com/facebookresearch/sapiens2/blob/main/docs/NORMAL.md)
- [Pointmap docs](https://github.com/facebookresearch/sapiens2/blob/main/docs/POINTMAP.md)
- [Pose docs](https://github.com/facebookresearch/sapiens2/blob/main/docs/POSE.md)
