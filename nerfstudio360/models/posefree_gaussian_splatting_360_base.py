import copy
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from os.path import join
from typing import Dict, List, Optional, Tuple, Type, Union
import cv2


import time
from PIL import Image
import imageio
import ipdb
import numpy as np
import math
import torch
import tqdm
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.camera_paths import get_interpolated_camera_path
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from gsplat360.optimizers import SelectiveAdam
from nerfstudio360.thirdparty.spherical_ssim import SphericalSSIM
from nerfstudio360.thirdparty.spherical_gncc import SphericalGNCC
from nerfstudio360.thirdparty.spherical_blur import SphericalBlur

from nerfstudio360.utils import camera_utils, colmap_free_utils, io_utils, pose_utils
from nerfstudio360.utils.camera_utils import build_unposed_camera, build_posed_camera
from nerfstudio360.utils.colmap_free_utils import GrowthState as GS
from nerfstudio360.utils.depth_utils import (
    generate_depth_sequence,
    generate_equir_depth_sequence,
    compute_aligned_depth,
)
from nerfstudio.utils.rich_utils import CONSOLE

json.encoder.FLOAT_REPR = lambda o: format(o, ".4f")

CameraOptimizer.selected_poses = colmap_free_utils.selected_poses
CameraOptimizer.all_poses = colmap_free_utils.all_poses
CameraOptimizer.update_poses = colmap_free_utils.update_poses


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=0.01, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


@dataclass
class PoseFreeGSplat360BaseModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: PoseFreeGSplat360BaseModel)

    camera_type: Literal["spherical", "perspective"] = "spherical"
    depth_type: Literal["midas", "unik3d"] = "unik3d"
    # depth_type: Literal["midas", "zoedepth", "depth_anywhere", "unik3d", "da2"] = "unik3d"

    # SCA Pose Solver
    spherical_aware: bool = True
    """Use spherical weighting in pose estimation."""
    consist_aware: bool = True
    """Use consistent masks in pose estimation."""
    internal_depth: bool = True
    """Use rendered internal depth maps to calculate camera poses"""
    match_views: int = 0
    """Use the last N frames to compute the pose of the latest frame"""

    # Pose Refinement
    refine_visited: bool = True
    """Refine poses of all visited cameras after coarse pose calulation"""
    consist_refine: bool = True
    """Use consistent masks in pose refinement"""

    # DIA Densify
    inlier_growth: bool = True
    """Use depth inlier merging"""
    align_depth: bool = True
    """Align monocular depths with rendered internal depths"""
    patch_filter: bool = True
    """Extract high-confidence depth inliers with patch level filter"""
    gncc_enhance: bool = True
    """Use GNCC or SSIM to identify high-confidence depth inliers"""
    outlier_remove: bool = True
    """Remove Gaussian outliers replaced by depth inliers"""
    outlier_reset: bool = True
    """Reset opacities of outliers that were not removed"""
    densify_views: int = 0
    """Use the last N frames to compute Gaussian inliers and outliers"""
    voxel_size: float = 0.01
    """Merge duplicate inliers using voxelization."""

    # Others
    blur_filter: bool = True
    """Use Gaussian blur smoothing on the obtained in-/consistent masks"""
    filter_sky: bool = True
    """Ignore sky areas with depth > 300 for camera optimization and gaussian growing"""
    sky_depth: float = 100.0

    # Gaussians regularization
    opacity_ratio: float = 0.01
    scale_ratio: float = 0.01
    phys_ratio: float = 0.1
    dist_ratio: float = 0.01

    background_color: Literal["random", "black", "white"] = "random"
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""
    """gaussian model to finetune scenes"""

    initial_gaussian_lr: float = 1.6e-4
    """optimize cameras during initial stage"""
    final_gaussian_lr: float = 1.6e-6
    """optimize cameras during finetune stage"""

    refine_camera_lr: float = 1e-3
    """optimize cameras during refinement stage"""
    joint_camera_lr: float = 1e-3
    """optimize cameras during joint optimization stage"""
    final_camera_lr: float = 5e-6
    """optimize cameras during finetune stage"""

    initial_interval: int = 1000
    """interval of initial monocular training"""
    refine_interval: int = 500
    """interval of camera pose refinement"""
    global_interval: int = 500
    """interval of global Gaussians optimization"""
    finetune_interval: int = 15000
    """interval of finetune training after growing of all frames"""

    early_stop: Optional[bool] = True
    """Stop after train and evaluate"""


class PoseFreeGSplat360BaseModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: PoseFreeGSplat360BaseModelConfig

    def __init__(
        self,
        *args,
        train_images=None,
        train_cameras: Cameras = None,
        eval_images=None,
        eval_cameras: Cameras = None,
        **kwargs,
    ):
        config = args[0]
        assert config.global_interval % config.refine_every == 0
        growth_steps = config.initial_interval + config.global_interval * (len(train_cameras) - 1)

        if config.camera_type == "spherical":
            assert (train_cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all() and (
                eval_cameras.camera_type == CameraType.EQUIRECTANGULAR.value
            ).all()
        elif config.camera_type == "perspective":
            assert (train_cameras.camera_type == CameraType.PERSPECTIVE.value).all() and (
                eval_cameras.camera_type == CameraType.PERSPECTIVE.value
            ).all()
        else:
            raise NotImplementedError
        self.spherical = config.camera_type == "spherical"
        self.groundtruth_cameras = copy.deepcopy(train_cameras)
        self.groundtruth_eval_cameras = copy.deepcopy(eval_cameras)
        self.train_images = copy.deepcopy(train_images)
        self.train_unposed_cameras = build_unposed_camera(train_cameras)  # [N, 3, 4]
        self.eval_images = copy.deepcopy(eval_images)
        self.eval_unposed_cameras = build_unposed_camera(eval_cameras)  # [N, 3, 4]

        del train_images, train_cameras, eval_images, eval_cameras

        if self.spherical:
            print("generating depth images with sphericl monocular depth estimation model ...")
            self.train_depths = generate_equir_depth_sequence(self.train_images, model=config.depth_type)
            self.eval_depths = generate_equir_depth_sequence(self.eval_images, model=config.depth_type)
        else:
            print("generating depth images with psepective monocular depth estimation model ...")
            self.train_depths = generate_depth_sequence(self.train_images, model=config.depth_type)
            self.eval_depths = generate_depth_sequence(self.eval_images, model=config.depth_type)

        self.train_backgrounds = self.train_depths.squeeze(-1) > config.sky_depth
        self.eval_backgrounds = self.eval_depths.squeeze(-1) > config.sky_depth

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=config.initial_gaussian_lr,
            lr_final=config.final_gaussian_lr,
            lr_delay_mult=0.01,
            lr_delay_steps=config.global_interval,
            max_steps=config.finetune_interval,
        )
        self.camera_scheduler_args = get_expon_lr_func(
            lr_init=config.joint_camera_lr,
            lr_final=config.final_camera_lr,
            lr_delay_mult=0.01,
            lr_delay_steps=config.global_interval,
            max_steps=config.finetune_interval,
        )

        super().__init__(*args, **kwargs)

    def populate_other_modules(self):
        self.crop_box = None
        self.set_background(torch.tensor([0.0, 0.0, 0.0]))
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=len(self.train_unposed_cameras), device="cpu"
        )
        self.eval_camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=len(self.eval_unposed_cameras), device="cpu"
        )

        self.spherical_ssim = SphericalSSIM(data_range=1.0, spherical=True)
        self.perspective_ssim = SphericalSSIM(data_range=1.0, spherical=False)
        self.spherical_gncc = SphericalGNCC(spherical=True)
        self.perspective_gncc = SphericalGNCC(spherical=False)
        self.spherical_blur = SphericalBlur(spherical=True)
        self.perspective_blur = SphericalBlur(spherical=False)

        # results
        self.placeholder = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("RPE_t", torch.zeros([1], dtype=torch.float32))
        self.register_buffer("RPE_r", torch.zeros([1], dtype=torch.float32))
        self.register_buffer("ATE", torch.zeros([1], dtype=torch.float32))
        self.register_buffer("RT", torch.zeros([1], dtype=torch.float32))
        self.start_time = time.perf_counter()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

        self.step = 0
        self.device_ready = False
        self.growth_stage = GS.DONE

        self.initial_length = self.config.initial_interval
        self.refine_interval = self.config.refine_interval
        self.global_interval = self.config.global_interval
        self.growth_interval = self.refine_interval + self.global_interval
        self.growth_length = self.growth_interval * (len(self.train_unposed_cameras) - 1)

        self.finetune_interval = self.config.finetune_interval
        self.finetune_length = self.config.global_interval + self.config.finetune_interval

        self.early_stop_at = self.initial_length + self.growth_length + self.finetune_length
        self.start_refine_at = self.initial_length
        self.stop_refine_at = self.initial_length + self.growth_length + self.global_interval

        self.start_time = time.perf_counter()

    def populate_modules(self):
        self.populate_other_modules()
        # TODO
        raise NotImplementedError

    def set_parameter(self, params="all"):

        self.selected_parameter = params
        assert params in ["none", "gaussians", "initial", "cameras", "eval_cameras", "all"]

        if params == "none":
            for name, param in self.gauss_params.named_parameters():
                param.requires_grad = False
            for name, param in self.camera_optimizer.named_parameters():
                param.requires_grad = False
            for name, param in self.eval_camera_optimizer.named_parameters():
                param.requires_grad = False
        if params == "gaussians":
            for name, param in self.gauss_params.named_parameters():
                param.requires_grad = True
            for name, param in self.camera_optimizer.named_parameters():
                param.requires_grad = False
            for name, param in self.eval_camera_optimizer.named_parameters():
                param.requires_grad = False
        if params == "initial":
            for name, param in self.gauss_params.named_parameters():
                if name == "means":
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            for name, param in self.camera_optimizer.named_parameters():
                param.requires_grad = False
            for name, param in self.eval_camera_optimizer.named_parameters():
                param.requires_grad = False
        if params == "cameras":
            for name, param in self.gauss_params.named_parameters():
                param.requires_grad = False
            for name, param in self.camera_optimizer.named_parameters():
                param.requires_grad = True
            for name, param in self.eval_camera_optimizer.named_parameters():
                param.requires_grad = False
        if params == "eval_cameras":
            for name, param in self.gauss_params.named_parameters():
                param.requires_grad = False
            for name, param in self.camera_optimizer.named_parameters():
                param.requires_grad = False
            for name, param in self.eval_camera_optimizer.named_parameters():
                param.requires_grad = True
        if params == "all":
            for name, param in self.gauss_params.named_parameters():
                param.requires_grad = True
            for name, param in self.camera_optimizer.named_parameters():
                param.requires_grad = True
            for name, param in self.eval_camera_optimizer.named_parameters():
                param.requires_grad = True

    def _train_view(
        self, index, background="black", ret_visible=False, query_values=None, gauss_confs=None, filter_sky=False
    ):
        camera = self.train_unposed_cameras[index : index + 1]
        c2w = self.camera_optimizer.selected_poses([index])
        return self.render_view(
            camera,
            c2w,
            background=background,
            ret_visible=ret_visible,
            query_values=query_values,
            gauss_confs=gauss_confs,
            filter_sky=filter_sky,
        )

    def _eval_view(self, index):
        camera = self.eval_unposed_cameras[index : index + 1]
        c2w = self.eval_camera_optimizer.selected_poses([index])
        return self.render_view(camera, c2w)

    @torch.no_grad()
    def _aligned_depth(self, this_index, last_index=None, next_index=None):

        this_camera = self.train_unposed_cameras[this_index : this_index + 1]
        this_c2w = self.camera_optimizer.selected_poses([this_index])
        this_depth = self._train_view(this_index)["depth"]
        consist = torch.ones_like(this_depth.squeeze(-1), dtype=torch.bool)
        if self.config.filter_sky:
            consist = torch.logical_and(consist, torch.logical_not(self.train_backgrounds[this_index]))

        if last_index is not None:
            last_camera = self.train_unposed_cameras[last_index : last_index + 1]
            last_c2w = self.camera_optimizer.selected_poses([last_index])
            last_depth = self._train_view(last_index)["depth"]
            last_consist = camera_utils.compute_consist_mask(
                this_camera, this_c2w, this_depth, last_camera, last_c2w, last_depth
            )
            consist = torch.logical_and(consist, last_consist)

        if next_index is not None:
            next_camera = self.train_unposed_cameras[next_index : next_index + 1]
            next_c2w = self.camera_optimizer.selected_poses([next_index])
            next_depth = self._train_view(next_index)["depth"]
            next_consist = camera_utils.compute_consist_mask(
                this_camera, this_c2w, this_depth, next_camera, next_c2w, next_depth
            )
            consist = torch.logical_and(consist, next_consist)

        aligned_depth = compute_aligned_depth(self.train_depths[this_index], this_depth, mask=consist)

        return aligned_depth

    def init_train_camera(self):
        idx = self.growth_index
        state_dict = copy.deepcopy(self.camera_optimizer.state_dict())
        pose = state_dict["pose_adjustment"][idx - 1 : idx]
        self.camera_optimizer.update_poses(pose, selected_indices=[idx])

    def register_train_camera(self, match_views=1):
        self.set_parameter("cameras")

        new_view = self.growth_index
        start_view = 0 if match_views == 0 else max(0, new_view - match_views)

        src_uids = list(range(start_view, new_view))
        tar_uids = [new_view]
        with torch.no_grad():
            src_cameras = self.train_unposed_cameras[start_view:new_view]
            src_poses = self.camera_optimizer.pose_adjustment.clone().detach()[start_view:new_view]
            src_images = self.train_images[start_view:new_view]
            src_depths = []
            for idx in range(start_view, new_view):
                if self.config.internal_depth:
                    src_depths.append(self._train_view(idx)["depth"].detach())
                else:
                    src_depths.append(self.train_depths[idx])

            src_depths = torch.stack(src_depths, dim=0)
            src_valids = torch.ones_like(src_images[..., 0], dtype=torch.bool)

            tar_cameras = self.train_unposed_cameras[new_view : new_view + 1]
            tar_images = self.train_images[new_view : new_view + 1]
            tar_valids = torch.ones_like(tar_images[..., 0], dtype=torch.bool)

            if self.config.consist_aware:

                if idx > 0:
                    src_valids *= self.twoview_consist(idx, idx - 1)
                    src_valids *= self.twoview_consist(idx, new_view - 1)
                if self.config.filter_sky:
                    src_valids *= torch.logical_not(self.train_backgrounds[start_view:new_view])
                    tar_valids *= torch.logical_not(self.train_backgrounds[new_view : new_view + 1])

        tar_poses = pose_utils.compute_camera_pose(
            src_cameras=src_cameras,
            src_uids=src_uids,
            src_poses=src_poses,
            src_images=src_images,
            src_depths=src_depths,
            src_valids=src_valids,
            tar_cameras=tar_cameras,
            tar_uids=tar_uids,
            tar_images=tar_images,
            tar_valids=tar_valids,
            config=self.config.camera_optimizer,
            spherical=self.spherical and self.config.spherical_aware,
        )

        self.camera_optimizer.update_poses(tar_poses, selected_indices=[self.growth_index])

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.int32)

    def set_crop(self, crop_box):
        # self.crop_box = crop_box
        self.crop_box = None

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    @torch.no_grad()
    def export_points(self, filename=None, export_cameras=True, mask=None):
        raise NotImplementedError

    @torch.no_grad()
    def export_train_views(self, savedir, max_index=None):
        os.makedirs(savedir, exist_ok=True)
        if max_index is None:
            max_index = len(self.train_unposed_cameras)
        for train_idx in tqdm.tqdm(range(max_index)):
            outputs = self._train_view(train_idx)
            io_utils.save_image(
                join(str(savedir), f"view_{train_idx:04d}_rgb.png"),
                torch.cat([self.train_images[train_idx], outputs["rgb"]], dim=1),
            )
            io_utils.save_depth(
                join(str(savedir), f"view_{train_idx:04d}_depth.png"),
                torch.cat([self.train_depths[train_idx], outputs["depth"]], dim=1),
            )

    @torch.no_grad()
    def render_video(self, savedir, steps: int = 15, fps: float = 30):
        c2ws = self.camera_optimizer.all_poses().clone().detach()  # [N, 3 ,4]
        posed_cameras = build_posed_camera(self.train_unposed_cameras, c2ws)
        cameras = get_interpolated_camera_path(cameras=posed_cameras, steps=steps, order_poses=True)
        cameras = cameras.to(c2ws.device)

        os.makedirs(join(savedir, "render"), exist_ok=True)

        for idx in range(len(cameras)):
            outputs = self.render_view(camera=cameras[idx : idx + 1])
            rgb = (outputs["rgb"] * 255).clip(min=0, max=255).type(torch.uint8).cpu().numpy()
            Image.fromarray(rgb).save(join(savedir, "render", f"frame-{idx:06d}.png"))

        import subprocess

        cmd = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-framerate",
            str(fps),
            "-i",
            join(savedir, "render", "frame-%06d.png"),  # 输入图片序列
            "-c:v",
            "libx264",  # x264 编码
            "-pix_fmt",
            "yuv420p",  # 兼容大部分播放器
            "-crf",
            "10",  # CRF=0 → 无损
            join(savedir, "render.mp4"),
        ]

        subprocess.run(cmd, check=True)

        import shutil

        shutil.rmtree(join(savedir, "render"), ignore_errors=True)

    @torch.no_grad()
    def plot_cameras(self, savedir, filename=None, num_cams=None):
        num_cams = num_cams if num_cams is not None else self.num_train_data
        c2ws_gt = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat([num_cams, 1, 1])
        c2ws_pred = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat([num_cams, 1, 1])
        selected_index = list(range(num_cams))
        c2ws_gt[:, :3, :] = self.groundtruth_cameras.camera_to_worlds[selected_index][:, :3].clone().detach()
        c2ws_pred[:, :3, :] = self.camera_optimizer.all_poses()[selected_index][:, :3].clone().detach()

        R_edit = torch.diag(torch.tensor([1, -1, -1], device=c2ws_gt.device, dtype=c2ws_gt.dtype))
        c2ws_gt[:, :3, :3] = c2ws_gt[:, :3, :3] @ R_edit
        c2ws_pred[:, :3, :3] = c2ws_pred[:, :3, :3] @ R_edit

        os.makedirs(savedir, exist_ok=True)
        try:
            rpe_t, rpe_r, ate, aligned_c2ws_gt, aligned_c2ws_pred = colmap_free_utils.align_cameras_and_worlds(
                c2ws_gt, c2ws_pred
            )

            self.register_buffer("RPE_t", rpe_t.view([1]).clone())
            self.register_buffer("RPE_r", rpe_r.view([1]).clone())
            self.register_buffer("ATE", ate.view([1]).clone())

            colmap_free_utils.plot_camera_pose(
                aligned_c2ws_gt.cpu().numpy(), aligned_c2ws_pred.cpu().numpy(), savedir, filename=f"{filename}.png"
            )

        except:
            c2ws_pred_dict = {}
            for idx in range(num_cams):
                c2ws_pred_dict[f"camera {idx:04d}"] = {
                    "c2w_R": self.camera_optimizer.selected_poses([idx]).squeeze()[:3, :3].cpu().tolist(),
                    "c2w_t": self.camera_optimizer.selected_poses([idx]).squeeze()[:3, 3].cpu().tolist(),
                }
            with open(join(savedir, f"{filename}-failed.json"), "w") as f:
                json.dump(c2ws_pred_dict, f, indent=4)

        with open(os.path.join(savedir, f"{filename}.json"), "w") as f:
            json.dump(
                {
                    "Points": self.num_points,
                    "RPE_t": self.RPE_t.item(),
                    "RPE_r": self.RPE_r.item(),
                    "ATE": self.ATE.item(),
                },
                f,
                indent=4,
            )

    @torch.no_grad()
    def plot_eval_cameras(self, savedir, filename=None, num_cams=None):
        num_cams = num_cams if num_cams is not None else len(self.eval_unposed_cameras)
        c2ws_gt = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat([num_cams, 1, 1])
        c2ws_pred = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat([num_cams, 1, 1])
        selected_index = list(range(num_cams))
        c2ws_gt[:, :3, :] = self.groundtruth_eval_cameras.camera_to_worlds[selected_index][:, :3].clone().detach()
        c2ws_pred[:, :3, :] = self.eval_camera_optimizer.all_poses()[selected_index][:, :3].clone().detach()

        R_edit = torch.diag(torch.tensor([1, -1, -1], device=c2ws_gt.device, dtype=c2ws_gt.dtype))
        c2ws_gt[:, :3, :3] = c2ws_gt[:, :3, :3] @ R_edit
        c2ws_pred[:, :3, :3] = c2ws_pred[:, :3, :3] @ R_edit

        os.makedirs(savedir, exist_ok=True)
        try:
            rpe_t, rpe_r, ate, aligned_c2ws_gt, aligned_c2ws_pred = colmap_free_utils.align_cameras_and_worlds(
                c2ws_gt, c2ws_pred
            )

            colmap_free_utils.plot_camera_pose(
                aligned_c2ws_gt.cpu().numpy(), aligned_c2ws_pred.cpu().numpy(), savedir, filename=f"{filename}.png"
            )

        except:
            c2ws_pred_dict = {}
            for idx in range(num_cams):
                c2ws_pred_dict[f"camera {idx:04d}"] = {
                    "c2w_R": self.camera_optimizer.selected_poses([idx]).squeeze()[:3, :3].cpu().tolist(),
                    "c2w_t": self.camera_optimizer.selected_poses([idx]).squeeze()[:3, 3].cpu().tolist(),
                }
            with open(join(savedir, f"{filename}-failed.json"), "w") as f:
                json.dump(c2ws_pred_dict, f, indent=4)

        with open(os.path.join(savedir, f"{filename}.json"), "w") as f:
            json.dump(
                {
                    "Points": self.num_points,
                    "RPE_t": rpe_t.item(),
                    "RPE_r": rpe_r.item(),
                    "ATE": ate.item(),
                },
                f,
                indent=4,
            )

    def load_status_to_device(self):
        if not self.device_ready:
            for key in self.growing_status.keys():
                self.growing_status[key] = self.growing_status[key].to(self.device)
            self.train_images = self.train_images.to(self.device)
            if self.config.filter_sky:
                self.train_backgrounds = self.train_backgrounds.to(self.device)
            self.train_depths = self.train_depths.to(self.device)
            self.train_unposed_cameras = self.train_unposed_cameras.to(self.device)
            self.groundtruth_cameras = self.groundtruth_cameras.to(self.device)
            self.groundtruth_eval_cameras = self.groundtruth_eval_cameras.to(self.device)
            self.eval_images = self.eval_images.to(self.device)
            if self.config.filter_sky:
                self.eval_backgrounds = self.eval_backgrounds.to(self.device)
            self.eval_depths = self.eval_depths.to(self.device)
            self.eval_unposed_cameras = self.eval_unposed_cameras.to(self.device)
            self.device_ready = True

    def merge_new_points(self, optimizers: Optimizers, new_points, new_colors):
        raise NotImplementedError

    def register_eval_cameras(self, training_callback_attributes: Optional[TrainingCallbackAttributes] = None):
        self.load_status_to_device()
        self.set_parameter("eval_cameras")
        assert self.growth_stage == GS.DONE and self.growth_step == 0

        selected_indices = (
            torch.linspace(0, len(self.train_unposed_cameras) - 1, len(self.eval_unposed_cameras)).long().tolist()
        )

        for idx in range(len(self.eval_unposed_cameras)):
            with torch.no_grad():
                src_index = selected_indices[idx]
                src_cameras = self.train_unposed_cameras[src_index : src_index + 1]
                src_uids = [src_index]
                src_poses = self.camera_optimizer.pose_adjustment.clone().detach()[src_index : src_index + 1]
                src_images = self.train_images[src_index : src_index + 1]
                src_depths = self._train_view(src_index)["depth"].detach().unsqueeze(0)
                src_valids = torch.ones_like(src_images[..., 0], dtype=torch.bool)
                if self.config.filter_sky:
                    src_valids *= torch.logical_not(self.train_backgrounds[src_index : src_index + 1])

                tar_cameras = self.eval_unposed_cameras[idx : idx + 1]
                tar_images = self.eval_images[idx : idx + 1]
                tar_valids = torch.ones_like(tar_images[..., 0], dtype=torch.bool)
                if self.config.filter_sky:
                    tar_valids *= torch.logical_not(self.eval_backgrounds[idx : idx + 1])

            tar_poses = pose_utils.compute_camera_pose(
                src_cameras=src_cameras,
                src_uids=src_uids,
                src_poses=src_poses,
                src_images=src_images,
                src_depths=src_depths,
                src_valids=src_valids,
                tar_cameras=tar_cameras,
                tar_uids=[idx],
                tar_images=tar_images,
                tar_valids=tar_valids,
                config=self.config.camera_optimizer,
            )

            # tar_poses = src_poses.clone().detach()

            self.eval_camera_optimizer.update_poses(tar_poses, selected_indices=[idx])

        camera_scheduler_args = get_expon_lr_func(
            lr_init=self.config.joint_camera_lr,
            lr_final=self.config.final_camera_lr,
            lr_delay_mult=0.01,
            max_steps=self.config.refine_interval,
        )

        for eval_index in tqdm.tqdm(range(len(self.eval_unposed_cameras))):
            if training_callback_attributes is not None:
                optimizer = training_callback_attributes.optimizers.optimizers["eval_camera_opt"]
            else:
                # optimizer = torch.optim.Adam([{"params": self.eval_camera_optimizer.parameters()}])
                optimizer = SelectiveAdam(
                    [{"params": self.eval_camera_optimizer.parameters()}],
                    eps=1e-15,
                    betas=(0.9, 0.999),
                    force_enable=True,
                )

            for it in range(self.config.refine_interval):

                if isinstance(optimizer, SelectiveAdam):
                    optimizer.set_index(
                        index=eval_index,
                        length=self.eval_camera_optimizer.num_cameras,
                        device=self.device,
                    )

                # lr_max = self.config.refine_camera_lr
                # lr_min = self.config.joint_camera_lr
                # lr = lr_min + (lr_max - lr_min) * max(0.5 * (math.cos(min(it, 100) / 100 * math.pi) + 1), 0)
                # for param_group in optimizer.param_groups:
                #     param_group["lr"] = lr

                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.config.refine_camera_lr

                optimizer.zero_grad()
                outputs = self._eval_view(eval_index)
                batch = {
                    "image": self.eval_images[eval_index],
                    "camera_idx": eval_index,
                    "image_idx": eval_index,
                }

                gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
                pred_img = outputs["rgb"]

                if "mask" in batch:
                    # batch["mask"] : [H, W, 1]
                    mask = self._downscale_if_required(batch["mask"])
                    mask = mask.to(self.device)
                    assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
                    gt_img = gt_img * mask
                    pred_img = pred_img * mask

                if "mask" in outputs:
                    # batch["mask"] : [H, W, 1]
                    mask = outputs["mask"].type(torch.float32)
                    assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
                    gt_img = gt_img * mask
                    pred_img = pred_img * mask

                loss = self._photo_loss(gt_img=gt_img, pred_img=pred_img, spherical=self.spherical)
                loss.backward()

                self.eval_camera_optimizer.pose_adjustment.grad.nan_to_num_(0, 0, 0)
                torch.nn.utils.clip_grad_value_(self.eval_camera_optimizer.pose_adjustment, clip_value=1e-3)
                optimizer.step()

            for it in range(self.config.refine_interval):

                if isinstance(optimizer, SelectiveAdam):
                    optimizer.set_index(
                        index=eval_index,
                        length=self.eval_camera_optimizer.num_cameras,
                        device=self.device,
                    )

                lr = camera_scheduler_args(it)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.zero_grad()
                outputs = self._eval_view(eval_index)
                batch = {
                    "image": self.eval_images[eval_index],
                    "camera_idx": eval_index,
                    "image_idx": eval_index,
                }

                gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
                pred_img = outputs["rgb"]

                if "mask" in batch:
                    # batch["mask"] : [H, W, 1]
                    mask = self._downscale_if_required(batch["mask"])
                    mask = mask.to(self.device)
                    assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
                    gt_img = gt_img * mask
                    pred_img = pred_img * mask

                if "mask" in outputs:
                    # batch["mask"] : [H, W, 1]
                    mask = outputs["mask"].type(torch.float32)
                    assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
                    gt_img = gt_img * mask
                    pred_img = pred_img * mask

                loss = self._photo_loss(gt_img=gt_img, pred_img=pred_img, spherical=self.spherical)
                loss.backward()

                self.eval_camera_optimizer.pose_adjustment.grad.nan_to_num_(0, 0, 0)
                torch.nn.utils.clip_grad_value_(self.eval_camera_optimizer.pose_adjustment, clip_value=1e-3)
                optimizer.step()

        torch.cuda.empty_cache()

    def config_growth_stage(self, step: int):
        # set growth states
        initial_length = self.initial_length
        refine_interval = self.refine_interval
        global_interval = self.global_interval
        growth_interval = self.growth_interval
        growth_length = self.growth_length
        finetune_length = self.finetune_length

        if step < initial_length:
            self.growth_stage = GS.INITIAL
            self.growth_index = 0
            self.growth_step = step
            self.train_index = self.growth_index

        elif (
            step >= initial_length
            and step < initial_length + growth_length
            and (step - initial_length) % growth_interval < refine_interval
        ):
            self.growth_stage = GS.RELATIVE
            self.growth_index = 1 + (step - initial_length) // growth_interval
            self.growth_step = (step - initial_length) % growth_interval
            if self.config.refine_visited:
                if random.random() < 0.7:
                    self.train_index = random.randint(self.growth_index // 2, self.growth_index)
                else:
                    self.train_index = random.randint(0, self.growth_index // 2)
            else:
                self.train_index = self.growth_index

        elif (
            step >= initial_length
            and step < initial_length + growth_length
            and (step - initial_length) % growth_interval >= refine_interval
        ):
            self.growth_stage = GS.GLOBAL
            self.growth_index = 1 + (step - initial_length) // growth_interval
            self.growth_step = (step - initial_length) % growth_interval - refine_interval
            if random.random() < 0.7:
                self.train_index = random.randint(self.growth_index // 2, self.growth_index)
            else:
                self.train_index = random.randint(0, self.growth_index // 2)
            # self.train_index = random.randint(0, self.growth_index)

        elif step >= initial_length + growth_length and step < initial_length + growth_length + finetune_length:
            self.growth_stage = GS.FINETUNE
            self.growth_index = self.num_train_data - 1
            self.growth_step = step - initial_length - growth_length
            if self.growth_step < self.global_interval:
                if random.random() < 0.7:
                    self.train_index = random.randint(self.growth_index // 2, self.growth_index)
                else:
                    self.train_index = random.randint(0, self.growth_index // 2)
            else:
                self.train_index = random.randint(0, self.num_train_data - 1)
            # self.train_index = random.randint(0, self.growth_index)

        else:
            self.growth_stage = GS.DONE
            self.growth_index = self.num_train_data - 1
            self.growth_step = step - initial_length - growth_length - finetune_length
            self.train_index = self.growth_step % (self.growth_index + 1)

    def update_lr(self, optimizers: Optimizers):

        if self.growth_stage == GS.INITIAL:
            for param_group in optimizers.optimizers["means"].param_groups:
                param_group["lr"] = self.config.initial_gaussian_lr
        if self.growth_stage == GS.GLOBAL:
            for param_group in optimizers.optimizers["means"].param_groups:
                param_group["lr"] = self.config.initial_gaussian_lr
        if self.growth_stage == GS.FINETUNE:
            lr = self.xyz_scheduler_args(self.growth_step)
            for param_group in optimizers.optimizers["means"].param_groups:
                param_group["lr"] = lr

        if self.growth_stage == GS.RELATIVE:
            for param_group in optimizers.optimizers["camera_opt"].param_groups:
                param_group["lr"] = self.config.refine_camera_lr

            # if not self.config.pnp_match:
            #     times = min(self.growth_step, self.config.refine_interval)
            #     lr_max = self.config.refine_camera_lr
            #     lr_min = self.config.joint_camera_lr
            #     lr = lr_min + (lr_max - lr_min) * max(
            #         0.5 * (math.cos(times / self.config.refine_interval * math.pi) + 1), 0
            #     )
            #     for param_group in optimizers.optimizers["camera_opt"].param_groups:
            #         param_group["lr"] = lr

        if self.growth_stage == GS.GLOBAL:
            for param_group in optimizers.optimizers["camera_opt"].param_groups:
                param_group["lr"] = self.config.joint_camera_lr
        if self.growth_stage == GS.FINETUNE:
            lr = self.camera_scheduler_args(self.growth_step)
            for param_group in optimizers.optimizers["camera_opt"].param_groups:
                param_group["lr"] = lr

    def before_train(self, training_callback_attributes: TrainingCallbackAttributes, step: int):
        self.step = step
        self.training_callback_attributes = training_callback_attributes
        self.scene = os.path.basename(str(training_callback_attributes.pipeline.datamanager.config.data))
        self.basedir = os.path.normpath(str(training_callback_attributes.trainer.base_dir))

        if not self.device_ready:
            self.load_status_to_device()

        self.config_growth_stage(step)
        self.update_lr(training_callback_attributes.optimizers)

        if isinstance(training_callback_attributes.optimizers.optimizers["camera_opt"], SelectiveAdam):
            training_callback_attributes.optimizers.optimizers["camera_opt"].set_index(
                index=self.train_index,
                length=self.camera_optimizer.num_cameras,
                device=self.device,
            )
        if isinstance(training_callback_attributes.optimizers.optimizers["eval_camera_opt"], SelectiveAdam):
            training_callback_attributes.optimizers.optimizers["eval_camera_opt"].set_index(
                index=[],
                length=self.eval_camera_optimizer.num_cameras,
                device=self.device,
            )

        if self.growth_stage == GS.INITIAL and self.growth_step == 0:
            self.set_parameter("cameras")
            self.initial_points(training_callback_attributes)
            self.set_parameter("initial")

        if self.growth_stage == GS.RELATIVE and self.growth_step == 0:
            self.set_parameter("cameras")
            self.register_train_camera(match_views=self.config.match_views)
            self.set_parameter("cameras")

        if self.growth_stage == GS.GLOBAL and self.growth_step == 0:
            self.set_parameter("all")
            if self.config.inlier_growth:
                self.inlier_growing(training_callback_attributes, last_index=self.growth_index - 1)
            self.set_parameter("all")

        if self.growth_stage == GS.FINETUNE and self.growth_step == 0:
            self.set_parameter("all")

        if self.growth_stage == GS.DONE and self.growth_step == 0:

            self.end_time = time.perf_counter()
            self.register_buffer("RT", torch.tensor([self.end_time - self.start_time], device=self.device))
            CONSOLE.print(f"full runtime: {(self.end_time - self.start_time)/60:.3f} mins")

            max_num_iterations = training_callback_attributes.trainer.config.max_num_iterations
            base_dir = training_callback_attributes.trainer.base_dir
            os.makedirs(os.path.normpath(str(base_dir / "results")), exist_ok=True)
            training_callback_attributes.trainer.save_checkpoint(max_num_iterations)

            self.plot_cameras(self.basedir, filename="pose")
            self.export_points(join(self.basedir, "points3D.ply"))
            if self.config.filter_sky:
                sky = self.growing_status["skys"] >= 0.8
                self.export_points(join(self.basedir, "points3D_nosky.ply"), mask=~sky)
            self.export_train_views(join(self.basedir, "train"))

            self.set_parameter("eval_cameras")
            self.register_eval_cameras(training_callback_attributes)
            self.plot_eval_cameras(self.basedir, filename="eval_pose")
            self.early_stop(training_callback_attributes)
            self.set_parameter("none")

    def after_train(self, training_callback_attributes: TrainingCallbackAttributes, step: int):
        raise NotImplementedError

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.before_train,
                args=[training_callback_attributes],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
                args=[training_callback_attributes],
            )
        )
        return cbs

    def early_stop(self, training_callback_attributes: TrainingCallbackAttributes, step: int = None):
        self.growth_stage = GS.DONE
        max_num_iterations = training_callback_attributes.trainer.config.max_num_iterations
        base_dir = training_callback_attributes.trainer.base_dir
        os.makedirs(os.path.normpath(str(base_dir / "results")), exist_ok=True)
        metrics = training_callback_attributes.pipeline.get_average_eval_image_metrics(output_path=base_dir / "results")
        metrics.update(
            {"RPE_t": self.RPE_t.item(), "RPE_r": self.RPE_r.item(), "ATE": self.ATE.item(), "RT": self.RT.item() / 60}
        )
        with open(str(base_dir / "results.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        training_callback_attributes.trainer.save_checkpoint(max_num_iterations)
        print(json.dumps(metrics, indent=4))
        exit(-1)

    def load_state_dict(self, state_dict, **kwargs):  # type: ignore
        self.growth_stage = GS.DONE
        self.growth_step = 0
        self.step = 1_000_000
        newp = state_dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        for key in ["RPE_t", "RPE_r", "ATE", "RT"]:
            if key in state_dict:
                state_dict[key] = state_dict[key].view([1])
        super().load_state_dict(state_dict, **kwargs)

    def _get_downscale_factor(self):
        return 1

    def resize_image(self, image: torch.Tensor, d: int):
        """
        Downscale images using the same 'area' method in opencv

        :param image shape [H, W, C]
        :param d downscale factor (must be 2, 4, 8, etc.)

        return downscaled image in shape [H//d, W//d, C]
        """
        import torch.nn.functional as tf

        image = image.to(torch.float32)
        weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
        return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        raise NotImplementedError

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return self.resize_image(image, d)
        return image

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def render_view(
        self,
        camera: Cameras,
        optional_c2w=None,
        background="black",
        ret_visible=False,
        query_values=None,
        gauss_confs=None,
        filter_sky=False,
    ) -> Dict[str, Union[torch.Tensor, List]]:
        raise NotImplementedError

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        assert camera.shape[0] == 1, "Only one camera at a time"

        if not self.device_ready:
            self.load_status_to_device()

        if self.training:
            optimized_c2w = self.camera_optimizer.selected_poses([self.train_index])
            background = "random"
        else:
            if camera.metadata is None:
                optimized_c2w = camera.camera_to_worlds
            elif "eval_idx" in camera.metadata:
                eval_index = camera.metadata["eval_idx"]
                optimized_c2w = self.eval_camera_optimizer.selected_poses([eval_index])
            else:
                optimized_c2w = camera.camera_to_worlds.to(self.device)
            background = "black"

        ret_visible = False
        gauss_confs = self._gauss_confs()
        query_values = None
        filter_sky = self.config.filter_sky and self.training

        rets = self.render_view(
            camera,
            optional_c2w=optimized_c2w,
            background=background,
            ret_visible=ret_visible,
            query_values=query_values,
            gauss_confs=gauss_confs,
            filter_sky=filter_sky,
        )

        return rets

    def get_depth_loss_dict(self, outputs, batch):
        weights = torch.tensor(1.0).to(self.device)
        depth_pred = outputs["depth"]
        depth_gt = self.train_depths[self.train_index].clone()

        if self.config.filter_sky:
            mask = torch.logical_not(self.train_backgrounds[self.train_index])
        else:
            mask = None
        depth_gt = compute_aligned_depth(depth_gt, depth_pred, mask=mask)

        delta = depth_pred.clip(min=1e-3) - depth_gt.clip(min=1e-3)
        depth_loss = (delta.abs() * weights).mean().nan_to_num(0.0, 0.0, 0.0)
        return {"depth": 0.01 * depth_loss}

    def _photo_loss(self, gt_img, pred_img, weights=None, spherical=False, ssim_lambda=0.2):
        Ll1 = (gt_img - pred_img).abs()
        if spherical:
            simloss = (1 - self.spherical_ssim(gt_img, pred_img)) / 2
        else:
            simloss = (1 - self.perspective_ssim(gt_img, pred_img)) / 2
        if weights is not None:
            Ll1 = Ll1 * weights
            simloss = simloss * weights
        return (1 - ssim_lambda) * Ll1.mean() + ssim_lambda * simloss.mean()

    @torch.no_grad()
    def _gauss_confs(self):
        if (
            self.config.consist_refine
            and self.training
            and self.growth_stage in [GS.INITIAL, GS.RELATIVE, GS.GLOBAL, GS.FINETUNE]
        ):
            confs = self.growing_status["confs"]
            counts = self.growing_status["counts"]
            if torch.all(counts == 0):
                return torch.zeros_like(confs)
            else:
                probs = torch.zeros_like(confs)
                valid_mask = counts > 0
                if self.config.filter_sky:
                    valid_mask *= self.growing_status["skys"] < 0.8
                valid_confs = confs[valid_mask]
                valid_probs = torch.softmax(-valid_confs, dim=0) * len(valid_confs)
                probs[valid_mask] = valid_probs
            return probs.contiguous()
        else:
            return None

    @torch.no_grad()
    def _consist_mask(
        self,
        this_index=None,
        start_index=None,
        end_index=None,
        this_depth=None,
        last_depth=None,
        next_depth=None,
        mode: Literal["and", "or"] = "and",
        # blur: bool = True,
    ):
        blur = self.config.blur_filter
        if this_index is None:
            this_index = self.train_index
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = self.growth_index

        this_camera = self.train_unposed_cameras[this_index : this_index + 1]
        this_c2w = self.camera_optimizer.selected_poses([this_index])
        if this_depth is None:
            this_depth = self._train_view(this_index)["depth"]
        consist = torch.ones_like(this_depth.squeeze(-1), dtype=torch.bool)

        last_idx = this_index - 1
        if last_idx >= start_index:
            last_camera = self.train_unposed_cameras[last_idx : last_idx + 1]
            last_c2w = self.camera_optimizer.selected_poses([last_idx])
            if last_depth is None:
                last_depth = self._train_view(last_idx)["depth"]
            last_consist = camera_utils.compute_consist_mask(
                this_camera, this_c2w, this_depth, last_camera, last_c2w, last_depth
            )
            consist = torch.logical_and(consist, last_consist)

        next_idx = this_index + 1
        if next_idx <= end_index:
            next_camera = self.train_unposed_cameras[next_idx : next_idx + 1]
            next_c2w = self.camera_optimizer.selected_poses([next_idx])
            if next_depth is None:
                next_depth = self._train_view(next_idx)["depth"]
            next_consist = camera_utils.compute_consist_mask(
                this_camera, this_c2w, this_depth, next_camera, next_c2w, next_depth
            )
            consist = torch.logical_and(consist, next_consist)

        if blur and self.spherical:
            consist = consist.unsqueeze(-1).type(torch.float32)
            consist = self.spherical_blur(consist).squeeze(-1) > 0.5
        if blur and not self.spherical:
            consist = consist.unsqueeze(-1).type(torch.float32)
            consist = self.perspective_blur(consist).squeeze(-1) > 0.5

        return consist

    @torch.no_grad()
    def twoview_consist(self, this_idx, ref_idx):
        blur = self.config.blur_filter
        this_camera = self.train_unposed_cameras[this_idx : this_idx + 1]
        this_c2w = self.camera_optimizer.selected_poses([this_idx])
        this_depth = self._train_view(this_idx)["depth"]

        ref_camera = self.train_unposed_cameras[ref_idx : ref_idx + 1]
        ref_c2w = self.camera_optimizer.selected_poses([ref_idx])
        ref_depth = self._train_view(ref_idx)["depth"]

        consist = camera_utils.compute_consist_mask(this_camera, this_c2w, this_depth, ref_camera, ref_c2w, ref_depth)
        if blur and self.spherical:
            consist = consist.unsqueeze(-1).type(torch.float32)
            consist = self.spherical_blur(consist).squeeze(-1) > 0.5
        if blur and not self.spherical:
            consist = consist.unsqueeze(-1).type(torch.float32)
            consist = self.perspective_blur(consist).squeeze(-1) > 0.5
        return consist

    @torch.no_grad()
    def _inconsist_mask(
        self, this_index=None, start_index=None, end_index=None, this_depth=None, last_depth=None, next_depth=None
    ):
        blur = self.config.blur_filter
        if this_index is None:
            this_index = self.train_index
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = self.growth_index

        this_camera = self.train_unposed_cameras[this_index : this_index + 1]
        this_c2w = self.camera_optimizer.selected_poses([this_index])
        if this_depth is None:
            this_depth = self._train_view(this_index)["depth"]

        inconsist = torch.ones_like(this_depth.squeeze(-1), dtype=torch.bool)

        last_idx = this_index - 1
        if last_idx >= start_index:
            last_camera = self.train_unposed_cameras[last_idx : last_idx + 1]
            last_c2w = self.camera_optimizer.selected_poses([last_idx])
            if last_depth is None:
                last_depth = self._train_view(last_idx)["depth"]
            last_consist = camera_utils.compute_consist_mask(
                this_camera, this_c2w, this_depth, last_camera, last_c2w, last_depth
            )
            inconsist = torch.logical_and(inconsist, torch.logical_not(last_consist))

        next_idx = this_index + 1
        if next_idx <= end_index:
            next_camera = self.train_unposed_cameras[next_idx : next_idx + 1]
            next_c2w = self.camera_optimizer.selected_poses([next_idx])
            if next_depth is None:
                next_depth = self._train_view(next_idx)["depth"]
            next_consist = camera_utils.compute_consist_mask(
                this_camera, this_c2w, this_depth, next_camera, next_c2w, next_depth
            )
            inconsist = torch.logical_and(inconsist, torch.logical_not(next_consist))

        if blur and self.spherical:
            inconsist = inconsist.unsqueeze(-1).type(torch.float32)
            inconsist = self.spherical_blur(inconsist).squeeze(-1) > 0.5
        if blur and not self.spherical:
            inconsist = inconsist.unsqueeze(-1).type(torch.float32)
            inconsist = self.perspective_blur(inconsist).squeeze(-1) > 0.5

        return inconsist

    @torch.no_grad()
    def _better_mask(
        self,
        this_index=None,
        start_index=None,
        end_index=None,
        this_depth=None,
        updated_depth=None,
        # blur: bool = True,
    ):
        blur = self.config.blur_filter
        if this_index is None:
            this_index = self.train_index
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = self.growth_index
        better = torch.ones_like(this_depth.squeeze(-1), dtype=torch.bool)

        if self.config.gncc_enhance:
            photo_func = self.spherical_gncc if self.spherical else self.perspective_gncc
        else:
            photo_func = self.spherical_ssim if self.spherical else self.perspective_ssim

        last_idx = this_index - 1
        if last_idx >= start_index:
            last_image, last_valid = camera_utils.compute_interp_image(
                this_camera=self.train_unposed_cameras[this_index : this_index + 1],
                this_c2w=self.camera_optimizer.selected_poses([this_index]),
                this_depth=this_depth,
                ref_camera=self.train_unposed_cameras[last_idx : last_idx + 1],
                ref_c2w=self.camera_optimizer.selected_poses([last_idx]),
                ref_image=self.train_images[last_idx],
            )
            last_updated_image, last_update_valid = camera_utils.compute_interp_image(
                this_camera=self.train_unposed_cameras[this_index : this_index + 1],
                this_c2w=self.camera_optimizer.selected_poses([this_index]),
                this_depth=updated_depth,
                ref_camera=self.train_unposed_cameras[last_idx : last_idx + 1],
                ref_c2w=self.camera_optimizer.selected_poses([last_idx]),
                ref_image=self.train_images[last_idx],
            )
            last_photo = photo_func(self.train_images[this_index], last_image).mean(dim=-1)
            last_updated_photo = photo_func(self.train_images[this_index], last_updated_image).mean(dim=-1)

            last_better = (last_updated_photo > last_photo) * last_valid * last_update_valid
            better = torch.logical_and(better, last_better)

        next_idx = this_index + 1
        if next_idx <= end_index:
            next_image, next_valid = camera_utils.compute_interp_image(
                this_camera=self.train_unposed_cameras[this_index : this_index + 1],
                this_c2w=self.camera_optimizer.selected_poses([this_index]),
                this_depth=this_depth,
                ref_camera=self.train_unposed_cameras[next_idx : next_idx + 1],
                ref_c2w=self.camera_optimizer.selected_poses([next_idx]),
                ref_image=self.train_images[next_idx],
            )
            next_updated_image, next_update_valid = camera_utils.compute_interp_image(
                this_camera=self.train_unposed_cameras[this_index : this_index + 1],
                this_c2w=self.camera_optimizer.selected_poses([this_index]),
                this_depth=updated_depth,
                ref_camera=self.train_unposed_cameras[next_idx : next_idx + 1],
                ref_c2w=self.camera_optimizer.selected_poses([next_idx]),
                ref_image=self.train_images[next_idx],
            )
            next_photo = photo_func(self.train_images[this_index], next_image).mean(dim=-1)
            next_updated_photo = photo_func(self.train_images[this_index], next_updated_image).mean(dim=-1)

            next_better = (next_updated_photo > next_photo) * next_valid * next_update_valid
            better = torch.logical_and(better, next_better)

        if blur and self.spherical:
            better = better.unsqueeze(-1).type(torch.float32)
            better = self.spherical_blur(better).squeeze(-1) > 0.5
        if blur and not self.spherical:
            better = better.unsqueeze(-1).type(torch.float32)
            better = self.perspective_blur(better).squeeze(-1) > 0.5

        return better

    @torch.no_grad()
    def _loss_consist_weight(self, outputs, batch=None):
        if not self.growth_stage == GS.RELATIVE or not self.config.consist_refine:
            return 1.0
        else:
            consist = self._consist_mask(self.train_index)
            weights = consist.unsqueeze(-1).type(torch.float32)
            if self.config.filter_sky:
                weights *= torch.logical_not(self.train_backgrounds[self.train_index]).unsqueeze(-1).type(torch.float32)
            return weights

    @torch.no_grad()
    def _loss_balance_weight(self, outputs, batch=None):
        if not self.spherical:
            return 1.0
        else:
            weights = torch.ones_like(self.train_images[self.train_index][..., 0:1], dtype=torch.float32)
            y = torch.arange(weights.shape[0], device=self.device, dtype=torch.float32)
            y = torch.sin((y + 0.5) / weights.shape[0] * torch.pi).clip(min=1e-3, max=1) * 0.8 + 0.2
            weights *= y.view(-1, 1, 1)
            return weights

    def get_loss_dict(self, outputs, batch=None, metrics_dict=None) -> Dict[str, torch.Tensor]:
        if self.growth_stage == GS.DONE:
            return {"placeholder": self.placeholder.mean() * 1e-3}

        batch = {
            "image": self.train_images[self.train_index],
            "camera_idx": self.train_index,
            "image_idx": self.train_index,
        }

        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        consist = self._loss_consist_weight(outputs, batch)
        balance = self._loss_balance_weight(outputs, batch)
        weights = consist * balance

        if "mask" in outputs:
            weights *= outputs["mask"].unsqueeze(-1).type(torch.float32)

        main_loss = self._photo_loss(gt_img=gt_img, pred_img=pred_img, weights=weights, spherical=self.spherical)

        # compute scale regularization loss
        use_scale_regularization = self.growth_stage in [GS.INITIAL, GS.GLOBAL, GS.FINETUNE]
        if use_scale_regularization and self.step % 10 == 0 and self.config.phys_ratio > 0:
            if self.config.filter_sky:
                scales = torch.exp(self.gauss_params["scales"][self.growing_status["skys"] < 0.8])
            else:
                scales = torch.exp(self.gauss_params["scales"])
            # scales = torch.exp(self.gauss_params["scales"])
            scales = torch.sort(scales, dim=-1).values
            scale_reg = torch.clip(scales[..., -1] / scales[..., 0].clip(min=1e-6) - 10.0, min=0)
            scale_reg = self.config.phys_ratio * scale_reg[scale_reg > 0].mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)
        scale_reg = scale_reg.nan_to_num_(0.0, 0.0, 0.0)

        # compute distortion regularization loss
        use_distortion_regularization = self.growth_stage in [GS.INITIAL, GS.GLOBAL, GS.FINETUNE]
        if use_distortion_regularization and self.config.dist_ratio > 0:
            depth_dist = balance * outputs["render_distort"] / outputs["depth"].clip(min=1e-3)
            depth_dist = self.config.dist_ratio * depth_dist.mean()
        else:
            depth_dist = torch.tensor(0.0).to(self.device)
        depth_dist = depth_dist.nan_to_num_(0.0, 0.0, 0.0)

        loss_dict = {
            "main_loss": main_loss,
            "scale_reg": scale_reg,
            "depth_dist": depth_dist,
        }

        use_depth_regularization = self.growth_stage == GS.INITIAL
        if use_depth_regularization:
            loss_dict.update(self.get_depth_loss_dict(outputs, batch))

        use_mcmc_regularization = self.growth_stage in [GS.INITIAL, GS.GLOBAL, GS.FINETUNE]
        if use_mcmc_regularization and self.config.opacity_ratio > 0:
            mcmc_opacity_reg = torch.sigmoid(self.gauss_params["opacities"])
            loss_dict["mcmc_opacity_reg"] = self.config.opacity_ratio * mcmc_opacity_reg.mean().nan_to_num_(
                0.0, 0.0, 0.0
            )
        if use_mcmc_regularization and self.config.scale_ratio > 0:
            if self.config.filter_sky:
                mcmc_scale_reg = torch.exp(self.gauss_params["scales"][self.growing_status["skys"] < 0.8])
            else:
                mcmc_scale_reg = torch.exp(self.gauss_params["scales"])
            loss_dict["mcmc_scale_reg"] = self.config.scale_ratio * mcmc_scale_reg.mean().nan_to_num_(0.0, 0.0, 0.0)

        if "camera_opt_regularizer" in loss_dict:
            del loss_dict["camera_opt_regularizer"]

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box=None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        # self.set_crop(obb_box)
        self.set_crop(None)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        if self.training:
            batch = {
                "image": self.train_images[self.train_index],
                "camera_idx": self.train_index,
                "image_idx": self.train_index,
            }
            mono_depth = self.train_depths[self.train_index]
        else:
            mono_depth = self.eval_depths[batch["camera_idx"]]

        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]

        images_dict = {"image_idx": batch["image_idx"], "gt": gt_rgb, "pred": predicted_rgb}

        if outputs.get("undist_rgb") is not None:
            images_dict["undist_pred"] = outputs["undist_rgb"]

        def get_adapt_plane(depth):
            depth = depth.ravel()
            far_plane, _ = torch.kthvalue(depth, int(depth.numel() * 0.95), dim=0)
            return far_plane.item() * 1.5

        if outputs.get("depth") is not None:
            images_dict["depth"] = colormaps.apply_depth_colormap(
                outputs["depth"],
                near_plane=0,
                far_plane=get_adapt_plane(outputs["depth"]),
                colormap_options=colormaps.ColormapOptions(colormap="turbo"),
            )

        if outputs.get("undist_depth") is not None:
            images_dict["undist_depth"] = colormaps.apply_depth_colormap(
                outputs["undist_depth"],
                near_plane=0,
                far_plane=get_adapt_plane(outputs["undist_depth"]),
                colormap_options=colormaps.ColormapOptions(colormap="turbo"),
            )

        if outputs.get("render_distort") is not None:
            images_dict["render_distort"] = colormaps.apply_depth_colormap(
                outputs["render_distort"],
                near_plane=0,
                far_plane=1,
                colormap_options=colormaps.ColormapOptions(colormap="turbo"),
            )

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
            "gaussians": float(outputs["gaussians"]),
        }  # type: ignore

        metrics_dict.update(
            {
                "RPE_t": self.RPE_t.item(),
                "RPE_r": self.RPE_r.item(),
                "ATE": self.ATE.item(),
                "RT": self.RT.item() / 60,
            }
        )

        if mono_depth is not None:
            images_dict["mono_depth"] = colormaps.apply_depth_colormap(
                mono_depth,
                far_plane=mono_depth.max().clip(min=1e-3).item(),
                colormap_options=colormaps.ColormapOptions(colormap="turbo"),
            )

        if self.config.filter_sky:
            if self.training:
                images_dict["background"] = self.train_backgrounds[self.train_index]
            else:
                images_dict["background"] = self.eval_backgrounds[batch["camera_idx"]]
            images_dict["background"] = images_dict["background"].type(torch.uint8) * 255

        return metrics_dict, images_dict
