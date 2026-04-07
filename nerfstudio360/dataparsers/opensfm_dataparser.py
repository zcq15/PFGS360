# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data parser for nerfstudio datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
    get_train_eval_split_all,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE
from collections import OrderedDict
import open3d
import cv2
import json
from nerfstudio360.utils import io_utils
import ipdb

MAX_AUTO_RESOLUTION = 6400


def box2str(box):
    return f"[[{box[0][0]:.3f}, {box[0][1]:.3f}, {box[0][2]:.3f}], [{box[1][0]:.3f}, {box[1][1]:.3f}, {box[1][2]:.3f}]]"


@dataclass
class OpenSfMDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: OpenSfMDataParser)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box near the poses' center."""
    pose_scale_factor: float = 1.0
    """How much to scale the camera origins by. This is conducted after auto scaling poses."""
    """
    After above three steps, the camera poses are in bounding box [-scale_factor + pose_center, scale_factor + pose_center],
    """
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    depth_unit_scale_factor: float = 1.0
    """The depth unit scale factor """
    eval_mode: Literal["all", "split"] = "split"
    """eval mode"""


@dataclass
class OpenSfMDataParser(DataParser):
    """Nerfstudio DatasetParser"""

    config: OpenSfMDataParserConfig
    downscale_factor: Optional[int] = None

    def _get_idx(self, filenames, splitlist):
        # sorted following the order in splitlist
        filenames = [os.path.basename(str(item)) for item in filenames]
        index_map = {value: idx for idx, value in enumerate(filenames)}
        indices = [index_map.get(fn) for fn in splitlist]
        return indices

    def _split_opensfm_subset(self, filenames, split):
        if self.config.eval_mode == "split":
            with open(str(self.config.data / "metadata.json"), "r") as f:
                metadata = json.load(f)
            if split == "train":
                splitlist = metadata["train"]
            elif split in ["val", "test"]:
                splitlist = metadata["test"]
            else:
                raise ValueError(f"Unknown dataparser split {split}")
            indices = self._get_idx(filenames=filenames, splitlist=splitlist)
        elif self.config.eval_mode == "all":
            CONSOLE.log(f"[red] loading all {len(filenames)} images for train and eval!")
            indices = list(range(len(filenames)))
        return indices

    def _parser_opensfm_assets(self):
        assert self.config.data.exists(), f"Data directory {str(self.config.data)} does not exist."
        assert (self.config.data / "opensfm" / "reconstruction.json").exists(), "Data {} does not exist.".format(
            self.config.data / "opensfm" / "reconstruction.json"
        )
        data_dir = self.config.data

        with open(str(data_dir / "opensfm" / "reconstruction.json"), "r") as f:
            meta = json.load(f)[0]

        image_filenames = []
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []
        poses = []

        # TODO: now, we assume that all cameras use the same camera spherical model
        assert len(meta["cameras"]) == 1

        for filename in sorted(list(meta["shots"].keys())):
            camera = meta["shots"][filename]["camera"]
            if not meta["cameras"][camera]["projection_type"] == "spherical":
                CONSOLE.print("camera of {} is not equirectangular, skip".format(filename))
                continue

            full_filename = self.config.data / "images" / filename
            image_filenames.append(full_filename)
            height.append(meta["cameras"][camera]["height"])
            width.append(meta["cameras"][camera]["width"])
            cx.append(float(meta["cameras"][camera]["width"]) / 2)
            cy.append(float(meta["cameras"][camera]["height"]) / 2)
            fx.append(float(meta["cameras"][camera]["height"]))
            fy.append(float(meta["cameras"][camera]["height"]))
            distort.append(camera_utils.get_distortion_params(k1=0.0, k2=0.0, k3=0.0, k4=0.0, p1=0.0, p2=0.0))

            R = cv2.Rodrigues(np.array(meta["shots"][filename]["rotation"], dtype=np.float32))[0]
            t = np.array(meta["shots"][filename]["translation"], dtype=np.float32).reshape(3, 1)
            # w2c = np.concatenate([R, t], 1)
            # w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]], dtype=np.float32)], 0)
            # c2w = np.linalg.inv(w2c)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R.T
            c2w[:3, 3:] = -R.T @ t
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[2, :] *= -1
            poses.append(torch.from_numpy(c2w).type(torch.float32))

        distort = torch.stack(distort, dim=0)
        poses = torch.stack(poses, dim=0)

        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([1, 0, 2]), :]
        applied_transform[2, :] *= -1
        applied_transform = torch.from_numpy(applied_transform).type(torch.float32)

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = self.config.pose_scale_factor
        pose_center = poses[:, :3, 3].mean(dim=0)
        if self.config.auto_scale_poses:
            pose_diff = poses[:, :3, 3] - pose_center.unsqueeze(0)
            scale_factor /= float(torch.max(pose_diff.norm(dim=-1)))
        poses[:, :3, 3] *= scale_factor
        pose_center *= scale_factor

        diff_pose = poses[:, :3, 3] - pose_center.unsqueeze(0)
        camera_box = torch.stack([diff_pose.amin(dim=0), diff_pose.amax(dim=0)], dim=0)
        CONSOLE.log(f"OpenSfM Camera Box: {box2str(camera_box)}")

        transform_matrix = transform_matrix @ torch.cat(
            [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)],
            dim=0,
        )

        metadata = self._load_opensfm_points(data_dir, transform_matrix, scale_factor)
        diff_xyz = metadata["points3D_xyz"] - pose_center.unsqueeze(0)
        diff_radius, _ = diff_xyz.norm(dim=-1).ravel().sort()
        radiis = diff_radius[int(0.95 * len(diff_radius))] * 3
        scene_box = SceneBox(aabb=torch.stack([pose_center - radiis, pose_center + radiis], dim=0).type(torch.float32))
        CONSOLE.log(f"OpenSfM Scene Box: {box2str(scene_box.aabb)}")

        metadata["transform_matrix"] = transform_matrix
        metadata["scale_factor"] = scale_factor
        metadata["scene_box"] = scene_box

        cameras = Cameras(
            fx=torch.tensor(fx, dtype=torch.float32).clone(),
            fy=torch.tensor(fy, dtype=torch.float32).clone(),
            cx=torch.tensor(cx, dtype=torch.float32).clone(),
            cy=torch.tensor(cy, dtype=torch.float32).clone(),
            distortion_params=distort.type(torch.float32).clone(),
            height=torch.tensor(height, dtype=torch.int32).clone(),
            width=torch.tensor(width, dtype=torch.int32).clone(),
            camera_to_worlds=poses[:, :3, :4].clone(),
            camera_type=CAMERA_MODEL_TO_TYPE["EQUIRECTANGULAR"],
            metadata={
                "camera_box": camera_box.view(1, 6).clone().tolist(),
                "scene_box": scene_box.aabb.view(1, 6).clone().tolist(),
            },
        )

        # TODO: now, we assume that all cameras use the same camera spherical model
        assert (cameras.height == cameras.height[0].item()).all() and (cameras.width == cameras.width[0].item()).all()

        return image_filenames, cameras, metadata

    def _generate_dataparser_outputs(self, split="train"):
        filenames, cameras, metadata = self._parser_opensfm_assets()
        indices = torch.tensor(self._split_opensfm_subset(filenames, split), dtype=torch.long)
        filenames = [filenames[i.item()] for i in indices]
        CONSOLE.print(f"loading images: {[os.path.basename(str((fn))) for fn in filenames]}")
        cameras = Cameras(
            fx=cameras.fx[indices].clone(),
            fy=cameras.fy[indices].clone(),
            cx=cameras.cx[indices].clone(),
            cy=cameras.cy[indices].clone(),
            distortion_params=cameras.distortion_params[indices].clone(),
            height=cameras.height[indices].clone(),
            width=cameras.width[indices].clone(),
            camera_to_worlds=cameras.camera_to_worlds[indices].clone(),
            camera_type=CAMERA_MODEL_TO_TYPE["EQUIRECTANGULAR"],
            metadata=cameras.metadata,
        )
        cameras.metadata["image_filenames"] = [os.path.basename(str(fn)) for fn in filenames]

        # self._export_scene(cameras, metadata, split)

        if split == "train" or self.config.eval_mode == "all":
            cameras.metadata["cam_idx"] = torch.arange(len(indices)).view(-1, 1).type(torch.long).clone()

        assert self.downscale_factor is None or self.downscale_factor == 1
        # cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        torch.cuda.empty_cache()

        # depth_filenames = []
        # if (self.config.data / "depths").exists():
        #     for fn in filenames:
        #         depth_filenames.append(self.config.data / "depths" / f"{fn.stem}_depth.png")

        background_filenames = []
        if (self.config.data / "backgrounds-grounded-sam-2").exists():
            for fn in filenames:
                background_filenames.append(
                    self.config.data / "backgrounds-grounded-sam-2" / f"{fn.stem}_background.png"
                )

        if "points3D_rgb" in metadata:
            metadata["points3D_rgb"] = (metadata["points3D_rgb"] * 255).clip(0, 255).type(torch.uint8)

        dataparser_outputs = DataparserOutputs(
            image_filenames=filenames,
            cameras=cameras,
            scene_box=metadata["scene_box"],
            mask_filenames=None,
            dataparser_scale=metadata["scale_factor"],
            dataparser_transform=metadata["transform_matrix"],
            metadata={
                # "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_filenames": None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "background_filenames": background_filenames if len(background_filenames) > 0 else None,
                **metadata,
            },
        )

        return dataparser_outputs

    def _load_opensfm_points(self, data_dir: Path, transform_matrix: torch.Tensor, scale_factor: float):
        points3D_xyz = []
        points3D_rgb = []
        with open(str(data_dir / "opensfm" / "reconstruction.json"), "r") as f:
            meta = json.load(f)[0]
        for _, item in meta["points"].items():
            points3D_xyz.append(item["coordinates"])
            points3D_rgb.append(item["color"])
        points3D_xyz = np.array(points3D_xyz, dtype=np.float32)
        points3D_rgb = np.array(points3D_rgb, dtype=np.float32) / 255.0
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points3D_xyz)
        pcd.colors = open3d.utility.Vector3dVector(points3D_rgb)
        open3d.io.write_point_cloud(str(data_dir / "opensfm" / "sparse.ply"), pcd)
        CONSOLE.log(
            "Loading {} points from {} ...".format(len(pcd.points), str(data_dir / "opensfm" / "reconstruction.json"))
        )
        points3D_xyz = torch.from_numpy(points3D_xyz).type(torch.float32)
        points3D_rgb = torch.from_numpy(points3D_rgb).type(torch.float32)
        points3D_xyz = (
            torch.cat((points3D_xyz, torch.ones_like(points3D_xyz[..., :1])), -1) @ transform_matrix.T[:4, :3]
        )
        points3D_xyz *= scale_factor

        return {"points3D_xyz": points3D_xyz, "points3D_rgb": points3D_rgb}

    def _export_scene(self, cameras: Cameras, metadata, split):
        points = metadata["points3D_xyz"].cpu()
        colors = metadata["points3D_rgb"].cpu()
        if colors.dtype == torch.uint8:
            colors = colors.type(torch.float32) / 255.0
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points.cpu().numpy())
        pcd.colors = open3d.utility.Vector3dVector(colors.cpu().numpy())

        c2ws = cameras.camera_to_worlds
        for idx in range(len(cameras)):
            pcd += io_utils.create_coordinate(c2w=c2ws[idx, :3, :4])
        pcd += io_utils.create_tracks(c2ws=c2ws)
        open3d.io.write_point_cloud(self.config.data / f"export_opensfm_dataparser_split_{split}.ply", pcd)
