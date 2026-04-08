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

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import open3d
import torch
from nerfstudio360.utils import io_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 6400


def box2str(box):
    return f"[[{box[0][0]:.3f}, {box[0][1]:.3f}, {box[0][2]:.3f}], [{box[1][0]:.3f}, {box[1][1]:.3f}, {box[1][2]:.3f}]]"


@dataclass
class OB3DDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: OB3DDataParser)
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
class OB3DDataParser(DataParser):
    """Nerfstudio DatasetParser"""

    config: OB3DDataParserConfig
    downscale_factor: Optional[int] = None

    @property
    def subdata(self):
        return self.config.data / self.config.trajectory_type

    def _read_camera(self, name):
        with open(str(self.subdata / "cameras" / f"{name}_cam.json"), "r") as f:
            meta = json.load(f)[0]
        width = torch.tensor(meta["width"], dtype=torch.int32)
        height = torch.tensor(meta["height"], dtype=torch.int32)
        fx = torch.tensor(meta["intrinsics"]["focal"], dtype=torch.float32)
        fy = torch.tensor(meta["intrinsics"]["focal"], dtype=torch.float32)
        cx = torch.tensor(meta["intrinsics"]["cx"], dtype=torch.float32)
        cy = torch.tensor(meta["intrinsics"]["cy"], dtype=torch.float32)
        distortion_params = torch.zeros([6], dtype=torch.float32)
        w2c_R = torch.tensor(meta["extrinsics"]["rotation"], dtype=torch.float32)
        w2c_t = torch.tensor(meta["extrinsics"]["translation"], dtype=torch.float32)

        camera_to_worlds = torch.eye(4, dtype=torch.float32)
        camera_to_worlds[:3, :3] = w2c_R.T
        camera_to_worlds[:3, 3] = -w2c_R.T @ w2c_t
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        camera_to_worlds[0:3, 1:3] *= -1
        camera_to_worlds = camera_to_worlds[np.array([1, 0, 2, 3]), :]
        camera_to_worlds[2, :] *= -1
        camera_to_worlds = camera_to_worlds[:3, :4]

        camera_type = CAMERA_MODEL_TO_TYPE["EQUIRECTANGULAR"]
        return Cameras(
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

    def _read_points(self, transform_matrix: torch.Tensor, scale_factor: float):
        pcd = open3d.io.read_point_cloud(str(self.subdata / "sparse" / "sparse.ply"))
        CONSOLE.log(
            "Loading {} points from {} ...".format(len(pcd.points), str(self.subdata / "sparse" / "sparse.ply"))
        )
        points3D_xyz = torch.from_numpy(np.asarray(pcd.points)).type(torch.float32)
        points3D_rgb = torch.from_numpy(np.asarray(pcd.colors)).type(torch.float32)
        points3D_xyz = (
            torch.cat((points3D_xyz, torch.ones_like(points3D_xyz[..., :1])), -1) @ transform_matrix.T[:4, :3]
        )
        points3D_xyz *= scale_factor
        return {"points3D_xyz": points3D_xyz, "points3D_rgb": points3D_rgb}

    def _parser_assets(self):
        image_filenames = sorted(os.listdir(str(self.subdata / "images")))
        filenames = [fn[:5] for fn in image_filenames]

        cameras_list = [self._read_camera(name) for name in filenames]
        width = torch.stack([camera.width for camera in cameras_list], dim=0)
        height = torch.stack([camera.height for camera in cameras_list], dim=0)
        fx = torch.stack([camera.fx for camera in cameras_list], dim=0)
        fy = torch.stack([camera.fy for camera in cameras_list], dim=0)
        cx = torch.stack([camera.cx for camera in cameras_list], dim=0)
        cy = torch.stack([camera.cy for camera in cameras_list], dim=0)
        distortion_params = torch.stack([camera.distortion_params for camera in cameras_list], dim=0)
        camera_to_worlds = torch.stack([camera.camera_to_worlds for camera in cameras_list], dim=0)
        camera_type = torch.stack([camera.camera_type for camera in cameras_list], dim=0)

        pose = torch.eye(4).unsqueeze(0).expand([len(filenames), -1, -1]).clone()
        pose[:, :3, :4] = camera_to_worlds.clone()
        pose, transform_matrix = camera_utils.auto_orient_and_center_poses(
            pose,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([1, 0, 2]), :]
        applied_transform[2, :] *= -1
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

        camera_to_worlds = pose[:, :3, :4].clone()
        cameras = Cameras(
            camera_to_worlds=camera_to_worlds,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            distortion_params=distortion_params,
            camera_type=camera_type,
            metadata={
                "camera_box": camera_box.view(1, 6).clone().tolist(),
                "scene_box": metadata["scene_box"].aabb.view(1, 6).clone().tolist(),
            },
        )

        return filenames, cameras, metadata

    def _split_subset(self, filenames, split):
        fn2idx = OrderedDict((filenames[idx], idx) for idx in range(len(filenames)))
        train_list = np.loadtxt(str(self.subdata / "train.txt"), dtype=int)
        test_list = np.loadtxt(str(self.subdata / "test.txt"), dtype=int)
        i_train = list(fn2idx[f"{fn:05d}"] for fn in train_list.tolist())
        i_eval = list(fn2idx[f"{fn:05d}"] for fn in test_list.tolist())
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        return indices

    def _generate_dataparser_outputs(self, split="train"):
        filenames, cameras, metadata = self._parser_assets()
        indices = torch.tensor(self._split_subset(filenames, split), dtype=torch.long)
        filenames = [self.subdata / "images" / f"{filenames[i.item()]}_rgb.png" for i in indices]
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

        # depth_filenames = []
        # if (self.config.data / "depths").exists():
        #     for fn in filenames:
        #         depth_filenames.append(self.config.data / "depths" / f"{fn.stem}_depth.png")

        background_filenames = []
        if (self.subdata / "backgrounds-grounded-sam-2").exists():
            path = str(self.subdata / "backgrounds-grounded-sam-2")
            CONSOLE.log(f"loading backgrounds from {path} ... ")
            for fn in filenames:
                background_filenames.append(self.subdata / "backgrounds-grounded-sam-2" / f"{fn.stem}_background.png")

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
        open3d.io.write_point_cloud(self.subdata / f"export_ob3d_dataperser_split_{split}.ply", pcd)
