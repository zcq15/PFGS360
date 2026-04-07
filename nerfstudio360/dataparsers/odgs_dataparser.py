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
import ipdb

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.rich_utils import CONSOLE
from collections import OrderedDict
import open3d
import cv2
import json
from nerfstudio360.utils import io_utils

MAX_AUTO_RESOLUTION = 6400


def box2str(box):
    return f"[[{box[0][0]:.3f}, {box[0][1]:.3f}, {box[0][2]:.3f}], [{box[1][0]:.3f}, {box[1][1]:.3f}, {box[1][2]:.3f}]]"


@dataclass
class ODGSDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: ODGSDataParser)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box near the poses' center."""
    pose_scale_factor: float = 1.0
    """How much to scale the camera origins by. This is conducted after auto scaling poses."""
    """
    After above three steps, the camera poses are in bounding box [-scale_factor + pose_center, scale_factor + pose_center],
    """
    depth_unit_scale_factor: float = 1.0
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    trajectory_type: Literal["Egocentric", "Non-Egocentric"] = "Egocentric"


@dataclass
class ODGSDataParser(DataParser):
    """Nerfstudio DatasetParser"""

    config: ODGSDataParserConfig
    downscale_factor: Optional[int] = None

    @property
    def data(self):
        return self.config.data

    def _read_cameras(self):
        cameras = []
        with open(str(self.data / f"data_extrinsics.json"), "r") as f:
            exts = json.load(f)["extrinsics"]
        exts = {item["key"]: item["value"] for item in exts}
        with open(str(self.data / f"data_views.json"), "r") as f:
            views = json.load(f)["views"]

        train_list = np.loadtxt(str(self.data / "train.txt"), dtype=str).tolist()
        test_list = np.loadtxt(str(self.data / "test.txt"), dtype=str).tolist()
        valid_list = train_list + test_list

        for item in views:
            assert item["key"] == item["value"]["ptr_wrapper"]["data"]["id_view"]
            assert item["key"] == item["value"]["ptr_wrapper"]["data"]["id_pose"]
            filename = item["value"]["ptr_wrapper"]["data"]["filename"]
            if not os.path.splitext(filename)[0] in valid_list:
                continue

            width = item["value"]["ptr_wrapper"]["data"]["width"]
            height = item["value"]["ptr_wrapper"]["data"]["height"]
            rotation = exts[item["key"]]["rotation"]
            center = exts[item["key"]]["center"]
            uid = item["key"]
            cameras.append(
                {
                    "uid": uid,
                    "filename": filename,
                    "width": width,
                    "height": height,
                    "rotation": rotation,
                    "center": center,
                }
            )

        cameras = sorted(cameras, key=lambda item: item["uid"])

        filenames = [item["filename"] for item in cameras]
        width = torch.tensor([item["width"] for item in cameras], dtype=torch.int32)
        height = torch.tensor([item["height"] for item in cameras], dtype=torch.int32)
        width = torch.tensor([item["width"] for item in cameras], dtype=torch.int32)
        height = torch.tensor([item["height"] for item in cameras], dtype=torch.int32)
        fx = torch.tensor([item["height"] for item in cameras], dtype=torch.float32)
        fy = torch.tensor([item["height"] for item in cameras], dtype=torch.float32)
        cx = torch.tensor([item["width"] / 2.0 for item in cameras], dtype=torch.float32)
        cy = torch.tensor([item["height"] / 2.0 for item in cameras], dtype=torch.float32)
        distortion_params = torch.zeros([len(cameras), 6], dtype=torch.float32)
        # In openMVG, Rcw * Ccw+tcw = 0, c2w_R = Rcw^T, C2W_T = tcw
        c2w_R = torch.tensor([item["rotation"] for item in cameras], dtype=torch.float32).permute(0, 2, 1)
        c2w_t = torch.tensor([item["center"] for item in cameras], dtype=torch.float32)

        camera_to_worlds = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(len(cameras), 1, 1)
        camera_to_worlds[:, :3, :3] = c2w_R
        camera_to_worlds[:, :3, 3] = c2w_t
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        camera_to_worlds[:, 0:3, 1:3] *= -1
        # camera_to_worlds = camera_to_worlds[:, np.array([1, 0, 2, 3]), :]
        # camera_to_worlds[:, 2, :] *= -1
        camera_to_worlds = camera_to_worlds[:, :3, :4]

        camera_type = torch.tensor([CAMERA_MODEL_TO_TYPE["EQUIRECTANGULAR"].value] * len(cameras))
        cameras = Cameras(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            camera_to_worlds=camera_to_worlds,
            camera_type=camera_type,
        )
        return filenames, cameras

    def _read_points(self, transform_matrix: torch.Tensor, scale_factor: float):
        pcd = open3d.io.read_point_cloud(str(self.data / "pcd.ply"))
        CONSOLE.log("Loading {} points from {} ...".format(len(pcd.points), str(self.data / "pcd.ply")))
        points3D_xyz = torch.from_numpy(np.asarray(pcd.points)).type(torch.float32)
        points3D_rgb = torch.from_numpy(np.asarray(pcd.colors)).type(torch.float32)
        points3D_xyz = (
            torch.cat((points3D_xyz, torch.ones_like(points3D_xyz[..., :1])), -1) @ transform_matrix.T[:4, :3]
        )
        points3D_xyz *= scale_factor
        return {"points3D_xyz": points3D_xyz, "points3D_rgb": points3D_rgb}

    def _parser_assets(self):

        filenames, cameras = self._read_cameras()

        pose = torch.eye(4).unsqueeze(0).expand([len(filenames), -1, -1]).clone()
        pose[:, :3, :4] = cameras.camera_to_worlds.clone()
        pose, transform_matrix = camera_utils.auto_orient_and_center_poses(
            pose,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )
        cameras.camera_to_worlds = pose[:, :3, :4].clone()

        applied_transform = np.eye(4)[:3, :]
        # applied_transform = applied_transform[np.array([1, 0, 2]), :]
        # applied_transform[2, :] *= -1
        applied_transform = torch.from_numpy(applied_transform).type(torch.float32)

        transform_matrix = transform_matrix @ torch.cat(
            [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)],
            dim=0,
        )

        metadata = self._read_points(transform_matrix=transform_matrix, scale_factor=1.0)
        radius = metadata["points3D_xyz"].norm(dim=-1).max().item()
        scene_box = torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]], dtype=torch.float32)
        camera_box = torch.stack([pose[:, :3, 3].amin(dim=0), pose[:, :3, 3].amax(dim=0)], dim=0)

        metadata.update(
            {
                "transform_matrix": transform_matrix,
                "scale_factor": 1.0,
                "scene_box": SceneBox(aabb=scene_box),
            }
        )

        cameras.metadata = {
            "camera_box": camera_box.view(1, 6).clone().tolist(),
            "scene_box": metadata["scene_box"].aabb.view(1, 6).clone().tolist(),
        }

        return filenames, cameras, metadata

    def _get_idx(self, filenames, splitlist):
        filelist = [os.path.splitext(os.path.basename(str(fn)))[0] for fn in filenames]
        fn2idx = {filelist[idx]: idx for idx in range(len(filelist))}
        splitlist = [os.path.splitext(os.path.basename(fn))[0] for fn in splitlist]
        indices = [fn2idx.get(fn, None) for fn in splitlist if fn in fn2idx]
        return indices

    def _split_subset(self, filenames, split):
        if split == "train":
            splitlist = self.data / "train.txt"
        elif split in ["val", "test"]:
            splitlist = self.data / "test.txt"
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        assert os.path.exists(str(splitlist))
        splitlist = np.loadtxt(str(splitlist), dtype=str).tolist()
        indices = self._get_idx(filenames=filenames, splitlist=splitlist)
        return indices

    def _generate_dataparser_outputs(self, split="train"):
        filenames, cameras, metadata = self._parser_assets()
        indices = torch.tensor(self._split_subset(filenames, split), dtype=torch.long)
        filenames = [self.data / "images" / f"{filenames[i]}" for i in indices]
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
            camera_type=cameras.camera_type[indices].clone(),
            metadata=cameras.metadata,
        )
        cameras.metadata["image_filenames"] = [os.path.basename(str(fn)) for fn in filenames]

        # self._export_scene(cameras, metadata, split)

        if split == "train":
            cameras.metadata["cam_idx"] = torch.arange(len(indices)).view(-1, 1).type(torch.long).clone()
        assert self.downscale_factor is None

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
                "depth_filenames": None,
                "background_filenames": background_filenames if len(background_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,
            },
        )

        return dataparser_outputs

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
        open3d.io.write_point_cloud(self.data / f"export_odgs_dataparser_split_{split}.ply", pcd)
