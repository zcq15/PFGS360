import functools
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union

import numpy as np
import open3d
import open3d as o3d
import torch
import torch_scatter
import tqdm
from nerfstudio360.models.posefree_gaussian_splatting_360_base import (
    PoseFreeGSplat360BaseModel,
    PoseFreeGSplat360BaseModelConfig,
)
from nerfstudio360.utils import camera_utils, colmap_free_utils, io_utils
from nerfstudio360.utils.camera_utils import build_posed_camera
from nerfstudio360.utils.colmap_free_utils import GrowthState as GS
from nerfstudio360.utils.depth_utils import compute_aligned_depth
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.splatfacto import RGB2SH, SH2RGB, get_viewmat, num_sh_bases, quat_to_rotmat, random_quat_tensor
from nerfstudio.utils.rich_utils import CONSOLE
from torch.nn import Parameter

from gsplat360 import rasterization
from gsplat360.optimizers import SelectiveAdam

json.encoder.FLOAT_REPR = lambda o: format(o, ".4f")

CameraOptimizer.selected_poses = colmap_free_utils.selected_poses
CameraOptimizer.all_poses = colmap_free_utils.all_poses


@dataclass
class PoseFreeGSplat360ModelConfig(PoseFreeGSplat360BaseModelConfig):
    _target: Type = field(default_factory=lambda: PoseFreeGSplat360Model)

    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""

    densify_grad_thresh: float = 0.00008  # nerfstudio: 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    densify_scale_thresh: float = 0.01  # nerfstudio: 0.01, cf-3dgs: 0.1
    """below this true-scale size, gaussians with large grads are *duplicated*, otherwise split"""
    split_screen_thresh: Optional[float] = None  # nerfstudio: 0.05, cf-3dgs: None
    """if a gaussian is more than this percent of screen space, split it"""
    split_pixel_thresh: Optional[int] = None  # nerfstudio: None, cf-3dgs: None
    """if a gaussian is more than this pixel size, split it"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    cull_screen_thresh: Optional[float] = None  # nerfstudio: 0.15, cf-3dgs: None
    """if a gaussian is more than this percent of screen space, cull it"""
    cull_pixel_thresh: Optional[int] = None  # nerfstudio: None, cf-3dgs: 20
    """if a gaussian is more than this pixel size, cull it"""
    cull_scale_thresh: Optional[float] = None  # nerfstudio: 0.5, cf-3dgs: 1.0
    """over this true-scale size, gaussians are pruned"""
    cull_alpha_thresh: float = 0.005  #  nerfstudio: 0.1, cf-3dgs: 0.005
    """over this true-scale size, gaussians are pruned"""


class PoseFreeGSplat360Model(PoseFreeGSplat360BaseModel):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: PoseFreeGSplat360ModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        self.populate_other_modules()
        means = torch.nn.Parameter(torch.zeros((0, 3)))
        scales = torch.nn.Parameter(torch.zeros((0, 3)))
        quats = torch.nn.Parameter(torch.zeros((0, 4)))
        features_dc = torch.nn.Parameter(torch.rand(0, 3))
        features_rest = torch.nn.Parameter(torch.zeros((0, num_sh_bases(self.config.sh_degree) - 1, 3)))
        opacities = torch.nn.Parameter(torch.logit(self.config.cull_alpha_thresh * 2.0 * torch.ones(0, 1)))

        self.growing_status = {
            "iters": torch.zeros(1, dtype=torch.int32),
            "grads": torch.zeros(0, dtype=torch.float32),
            "counts": torch.zeros(0, dtype=torch.float32),
            "maxsize": torch.zeros(0, dtype=torch.float32),
        }

        if self.config.consist_aware:
            self.growing_status["confs"] = torch.zeros(0, dtype=torch.float32)
        if self.config.filter_sky:
            self.growing_status["times"] = torch.zeros(0, dtype=torch.int32)
            self.growing_status["skys"] = torch.zeros(0, dtype=torch.float32)

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        if self.config.sh_degree > 0:
            return self.features_dc
        else:
            return RGB2SH(torch.sigmoid(self.features_dc))

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        gps["camera_opt"] = list(self.camera_optimizer.parameters())
        gps["eval_camera_opt"] = list(self.eval_camera_optimizer.parameters())
        gps["placeholder"] = [self.placeholder]
        return gps

    @torch.no_grad()
    def export_points(self, filename=None, export_cameras=True, mask=None):
        points = self.gauss_params["means"].clone().detach()
        colors = SH2RGB(self.gauss_params["features_dc"].clone().detach())
        opacities = torch.sigmoid(self.gauss_params["opacities"].clone().detach())
        valid = opacities.squeeze(-1) > self.config.cull_alpha_thresh
        if mask is not None:
            valid *= mask
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points[valid].cpu().numpy())
        pcd.colors = open3d.utility.Vector3dVector(colors[valid].clip(min=0, max=1).cpu().numpy())
        if export_cameras:
            c2ws = self.camera_optimizer.all_poses()
            for idx in range(len(self.train_unposed_cameras)):
                pcd += io_utils.create_coordinate(c2w=c2ws[idx, :3, :4])
            pcd += io_utils.create_tracks(c2ws=c2ws)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        open3d.io.write_point_cloud(filename, pcd)

    def remove_from_optim_pro(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim_pro(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim_pro(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim_pro(self, optimizer, new_params, num_new_anchors):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (num_new_anchors,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][:1]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][:1]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim_pro(self, optimizers, num_new_anchors):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim_pro(optimizers.optimizers[group], param, num_new_anchors)

    def merge_new_points(self, optimizers: Optimizers, new_points, new_colors, new_alpha: float = None):

        num_points = new_points.shape[0]
        if num_points < 10:
            return None

        voxel_size = self.config.voxel_size

        if voxel_size > 0:

            new_voxel_grid, inverse_index = torch.unique(
                torch.round(new_points / voxel_size).type(torch.int32), dim=0, return_inverse=True
            )
            last_voxel_grid = torch.unique(torch.round(self.means.detach() / voxel_size).type(torch.int32), dim=0)
            grids = torch.cat([new_voxel_grid, last_voxel_grid], dim=0)
            new_colors = torch_scatter.scatter_mean(new_colors, index=inverse_index, dim=0)

            if new_voxel_grid.shape[0] < 100:
                return None

            try:
                # compute distance
                voxels = grids.type(torch.float32) * voxel_size
                distances, _ = self.k_nearest_sklearn(voxels.cpu(), 3)
                distances = torch.from_numpy(distances[: new_voxel_grid.shape[0]]).to(self.device)
                distances = distances.mean(dim=-1).clip(max=1.0)
            except:
                return None

            # compute selected index

            _, scatter_index, grid_counts = torch.unique(grids, return_inverse=True, return_counts=True, dim=0)
            selected_mask = torch.index_select(grid_counts == 1, index=scatter_index[: new_voxel_grid.shape[0]], dim=0)
            selected_points = new_voxel_grid[selected_mask].contiguous().type(torch.float32) * voxel_size
            selected_colors = new_colors[selected_mask]
            selected_dist = distances[selected_mask]

            # compute new params
            num_selected_anchors = selected_mask.type(torch.uint32).sum().item()
            if num_selected_anchors == 0:
                return None
            else:
                CONSOLE.print(
                    f"merge {num_selected_anchors} gaussians with {num_points} points, from {self.num_points} to {self.num_points+num_selected_anchors}"
                )
        else:
            try:
                # compute distance
                distances, _ = self.k_nearest_sklearn(new_points.cpu(), 3)
                distances = torch.from_numpy(distances).to(self.device)
                distances = distances.mean(dim=-1).clip(max=1.0)
            except:
                return None

            num_selected_points = new_points.shape[0]
            selected_points = new_points
            selected_colors = new_colors
            selected_dist = distances
            CONSOLE.print(
                f"merge {num_selected_points} gaussians with {num_selected_points}  points, from {self.num_points} to {self.num_points+num_selected_points}"
            )

        means = torch.nn.Parameter(selected_points)
        scales = torch.nn.Parameter(torch.log(selected_dist.unsqueeze(-1).repeat(1, 3)))
        quats = torch.nn.Parameter(random_quat_tensor(means.shape[0]).to(self.device))
        dim_sh = num_sh_bases(self.config.sh_degree)
        shs = torch.zeros((means.shape[0], dim_sh, 3)).type(torch.float32).to(self.device)
        if self.config.sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(selected_colors)
            shs[:, 1:, 3:] = 0.0
        else:
            CONSOLE.print("use color only optimization with sigmoid activation")
            shs[:, 0, :3] = torch.logit(selected_colors, eps=1e-10)
        features_dc = torch.nn.Parameter(shs[:, 0, :].to(self.device))
        features_rest = torch.nn.Parameter(shs[:, 1:, :].to(self.device))

        if new_alpha is None:
            opacities = torch.nn.Parameter(
                torch.logit(2.0 * self.config.cull_alpha_thresh * torch.ones(means.shape[0], 1)).to(self.device)
            )
        else:
            assert new_alpha > 0 and new_alpha < 1
            opacities = torch.nn.Parameter(torch.logit(new_alpha * torch.ones(means.shape[0], 1)).to(self.device))

        new_gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(
                torch.cat([param.detach(), new_gauss_params[name].detach()], dim=0)
            )
        self.dup_in_all_optim_pro(optimizers, means.shape[0])

        for key in ["grads", "counts", "maxsize", "confs", "times", "skys"]:
            if key in self.growing_status:
                self.growing_status[key] = torch.cat(
                    [
                        self.growing_status[key],
                        torch.zeros([means.shape[0]], dtype=self.growing_status[key].dtype, device=self.device),
                    ],
                    dim=0,
                )

    def initial_points(self, training_callback_attributes: TrainingCallbackAttributes):
        assert self.growth_stage in [GS.INITIAL]

        c2w = self.camera_optimizer.selected_poses([0]).clone().detach()
        camera = build_posed_camera(self.train_unposed_cameras[0:1], c2w)
        rays = camera.generate_rays(0, keep_shape=True)
        seed_points = self.train_depths[0] * rays.directions * rays.metadata["directions_norm"] + rays.origins
        seed_colors = self.train_images[0]

        seed_points = seed_points.view(-1, 3)
        seed_colors = seed_colors.view(-1, 3)

        if self.config.voxel_size > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(seed_points.clone().detach().cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(seed_colors.clone().detach().cpu().numpy())
            pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=self.config.voxel_size)
            seed_points = torch.from_numpy(np.array(pcd.points)).type(torch.float32).to(self.device)
            seed_colors = torch.from_numpy(np.array(pcd.colors)).type(torch.float32).to(self.device)

        self.merge_new_points(
            training_callback_attributes.optimizers, seed_points.clone().detach(), seed_colors.clone().detach()
        )

    def inlier_growing(self, training_callback_attributes: TrainingCallbackAttributes, last_index: int):
        if last_index <= 0:
            return

        num_points_before = self.num_points

        start_index = 0 if self.config.densify_views == 0 else max(0, last_index - self.config.densify_views + 1)
        end_index = last_index + 1
        indices = list(range(start_index, end_index))

        # compute render depth
        render_depths = {}
        for index in indices:
            with torch.no_grad():
                render_depths[index] = self._train_view(index)["depth"]

        # compute consist mask without sky
        render_consists = {}
        for index in indices:
            _consist = self._consist_mask(
                this_index=index,
                start_index=start_index,
                end_index=end_index,
                this_depth=render_depths[index],
                last_depth=render_depths.get(index - 1),
                next_depth=render_depths.get(index + 1),
            )
            render_consists[index] = _consist

        if self.config.filter_sky:
            for index in indices:
                render_consists[index] = torch.logical_and(
                    render_consists[index], torch.logical_not(self.train_backgrounds[index])
                )

        # compute inconsist mask with sky
        render_inconsists = {}
        for index in indices:
            _inconsist = self._inconsist_mask(
                this_index=index,
                start_index=start_index,
                end_index=end_index,
                this_depth=render_depths[index],
                last_depth=render_depths.get(index - 1),
                next_depth=render_depths.get(index + 1),
            )
            render_inconsists[index] = _inconsist

        if self.config.filter_sky:
            for index in indices:
                render_inconsists[index] = torch.logical_and(
                    render_inconsists[index], torch.logical_not(self.train_backgrounds[index])
                )
        # compute aligned monocular depth
        aligned_depth = {}
        for index in indices:
            if self.config.align_depth:
                aligned_depth[index] = compute_aligned_depth(
                    self.train_depths[index], render_depths[index], mask=render_consists[index]
                )
            else:
                aligned_depth[index] = self.train_depths[index].clone()

        # compute consist points without sky
        mono_inliers = {k: v.clone().detach() for k, v in render_inconsists.items()}
        for index in indices:
            _consist = self._consist_mask(
                this_index=index,
                start_index=start_index,
                end_index=end_index,
                this_depth=aligned_depth[index],
                last_depth=aligned_depth.get(index - 1),
                next_depth=aligned_depth.get(index + 1),
            )
            mono_inliers[index] = torch.logical_and(mono_inliers[index], _consist)

        if self.config.patch_filter:
            for index in indices:
                _better = self._better_mask(
                    this_index=index,
                    start_index=start_index,
                    end_index=end_index,
                    this_depth=render_depths[index],
                    updated_depth=aligned_depth[index],
                )
                mono_inliers[index] = torch.logical_and(mono_inliers[index], _better)

        if self.config.filter_sky:
            for index in indices:
                mono_inliers[index] = torch.logical_and(
                    mono_inliers[index], torch.logical_not(self.train_backgrounds[index])
                )

        # compute inconsist points
        inc_hits = torch.zeros([self.num_points], dtype=torch.int32, device=self.device)
        inlier_hits = torch.zeros([self.num_points], dtype=torch.int32, device=self.device)
        inc_probs = torch.zeros([self.num_points], dtype=torch.float32, device=self.device)
        inlier_probs = torch.zeros([self.num_points], dtype=torch.float32, device=self.device)
        resp_probs = torch.zeros([self.num_points], dtype=torch.float32, device=self.device)
        for index in indices:
            query_values = torch.stack([render_inconsists[index], mono_inliers[index]], dim=-1)
            with torch.no_grad():
                rets = self._train_view(
                    index=index, query_values=query_values.unsqueeze(0).type(torch.float32), ret_visible=True
                )
            visible = torch.logical_and(rets["radii"] > 0, rets["accum_visible"] > 0)
            inc_probs[visible] += rets["query_answers"][visible][:, 0]
            inlier_probs[visible] += rets["query_answers"][visible][:, 1]
            resp_probs[visible] += rets["accum_visible"][visible]
            inc_hits[visible] += (rets["query_answers"][visible][:, 0] / rets["accum_visible"][visible] > 0.8).type(
                torch.int32
            )
            inlier_hits[visible] += (rets["query_answers"][visible][:, 1] / rets["accum_visible"][visible] > 0.8).type(
                torch.int32
            )

        inc_probs = torch.where(resp_probs > 0, inc_probs / resp_probs.clip(min=1e-12), torch.zeros_like(inc_probs))
        inlier_probs = torch.where(
            resp_probs > 0, inlier_probs / resp_probs.clip(min=1e-12), torch.zeros_like(inlier_probs)
        )

        # culls = inlier_probs > 0.8
        # resets = torch.logical_and(inc_probs > 0.8, torch.logical_not(culls))
        culls = inlier_hits > 0
        resets = torch.logical_and(inc_hits > 0, torch.logical_not(culls))

        num_culls = culls.type(torch.int32).sum().item()
        num_resets = resets.type(torch.int32).sum().item()

        # reset
        if num_resets >= 100 and self.config.outlier_reset:
            self.growing_status["counts"][resets] = 0
            optimizers = training_callback_attributes.optimizers
            reset_value = self.config.cull_alpha_thresh * 2.0
            reset_value = torch.logit(torch.tensor(reset_value, device=self.device)).item()
            reseted_opacities = torch.clamp(self.opacities.data[resets], max=reset_value)
            self.opacities.data[resets] = reseted_opacities
            self.opacities.data.contiguous()
            # reset the exp of optimizer
            optim = optimizers.optimizers["opacities"]
            param = optim.param_groups[0]["params"][0]
            param_state = optim.state[param]
            param_state["exp_avg"] = torch.where(
                resets.unsqueeze(-1).expand_as(param_state["exp_avg"]),
                torch.zeros_like(param_state["exp_avg"]),
                param_state["exp_avg"],
            )
            param_state["exp_avg_sq"] = torch.where(
                resets.unsqueeze(-1).expand_as(param_state["exp_avg_sq"]),
                torch.zeros_like(param_state["exp_avg_sq"]),
                param_state["exp_avg_sq"],
            )

        # remove
        if num_culls >= 100 and self.config.outlier_remove:
            for name, param in self.gauss_params.items():
                self.gauss_params[name] = torch.nn.Parameter(param[~culls])
            for key in ["grads", "counts", "maxsize", "confs", "times", "skys"]:
                if key in self.growing_status:
                    self.growing_status[key] = self.growing_status[key][~culls].clone().detach()
            self.remove_from_all_optim(training_callback_attributes.optimizers, culls)

        # merge
        consist_pcd = o3d.geometry.PointCloud()
        with torch.no_grad():
            for index in indices:
                c2w = self.camera_optimizer.selected_poses([index]).clone().detach()
                camera = camera_utils.build_posed_camera(self.train_unposed_cameras[index : index + 1], c2w)
                rays = camera.generate_rays(0, keep_shape=True)
                seed_points = aligned_depth[index] * rays.directions * rays.metadata["directions_norm"] + rays.origins
                seed_colors = self.train_images[index]

                selected = mono_inliers[index]
                if selected.type(torch.int32).sum().item() >= 100:
                    seed_points = seed_points[selected]
                    seed_colors = seed_colors[selected]
                    _pcd = o3d.geometry.PointCloud()
                    _pcd.points = o3d.utility.Vector3dVector(seed_points.clone().detach().cpu().numpy())
                    _pcd.colors = o3d.utility.Vector3dVector(seed_colors.clone().detach().cpu().numpy())
                    consist_pcd += _pcd

        num_inliers = len(consist_pcd.points)
        if num_inliers >= 100:
            if self.config.voxel_size > 0:
                consist_pcd = o3d.geometry.PointCloud.voxel_down_sample(consist_pcd, voxel_size=self.config.voxel_size)
            seed_points = torch.from_numpy(np.array(consist_pcd.points)).type(torch.float32).to(self.device)
            seed_colors = torch.from_numpy(np.array(consist_pcd.colors)).type(torch.float32).to(self.device)
            self.merge_new_points(
                training_callback_attributes.optimizers,
                seed_points.clone().detach(),
                seed_colors.clone().detach(),
            )

        del render_depths, render_consists, render_inconsists, aligned_depth, mono_inliers
        torch.cuda.empty_cache()

        num_points_after = self.num_points
        CONSOLE.print(
            f"culls {num_culls}, resets {num_resets}, merges {num_inliers}, after inlier growth and reseting, from {num_points_before} to {num_points_after}"
        )

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.gauss_params["scales"][split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.gauss_params["scales"][split_mask]) / size_fac).repeat(samps, 1)
        self.gauss_params["scales"][split_mask] = torch.log(
            torch.exp(self.gauss_params["scales"][split_mask]) / size_fac
        )
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        # cull huge ones
        if self.config.cull_scale_thresh is not None:
            culls = culls | (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
        if self.config.cull_screen_thresh is not None and self.growing_status["maxsize"] is not None:
            culls = culls | (self.growing_status["maxsize"] > self.config.cull_screen_thresh).squeeze()
        if self.config.cull_pixel_thresh is not None and self.growing_status["maxsize"] is not None:
            culls = (
                culls
                | (
                    self.growing_status["maxsize"] * min(self.growing_status["width"], self.growing_status["height"])
                    > self.config.cull_pixel_thresh
                ).squeeze()
            )
        with torch.no_grad():
            ood = torch.zeros([self.num_points], dtype=torch.bool, device=self.device)
            for idx in range(self.growth_index + 1):
                c2w = self.camera_optimizer.selected_poses([idx])
                dist = (self.means - c2w.squeeze()[:3, 3].view(1, 3)).norm(dim=-1)
                ood = ood | (dist > 1e5)
            culls = culls | ood

        # print(f"cull {culls.sum()} points")
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        for key in ["grads", "counts", "maxsize", "confs", "times", "skys"]:
            if key in self.growing_status:
                self.growing_status[key] = self.growing_status[key][~culls].clone().detach()

        return culls

    @torch.no_grad()
    def update_statis(self):
        # keep track of a moving average of grad norms
        visible_mask = (self.growing_status["radii"] > 0).flatten()
        grads = self.growing_status["xys"].absgrad.clone().detach()[0]

        grads[..., 0] *= self.growing_status["width"] / 2.0
        grads[..., 1] *= self.growing_status["height"] / 2.0
        grads = grads.norm(dim=-1).nan_to_num(0, 0, 0)

        if self.config.consist_aware:
            last_confs = self.growing_status["confs"].clone()
            new_confs = torch.where(self.growing_status["counts"] > 0, last_confs * 0.99 + 0.01 * grads, grads)
            self.growing_status["confs"][visible_mask] = new_confs[visible_mask]

        if self.config.filter_sky:
            resp_mask = self.growing_status["resp_visible"] > 0
            resp_skys = self.growing_status["resp_skys"][resp_mask] / self.growing_status["resp_visible"][resp_mask]
            last_skys = self.growing_status["skys"].clone()[resp_mask]
            new_skys = torch.where(
                self.growing_status["times"][resp_mask] > 0,
                last_skys * 0.99 + 0.01 * resp_skys.type(torch.float32),
                resp_skys.type(torch.float32),
            )
            self.growing_status["times"][resp_mask] += 1
            self.growing_status["skys"][resp_mask] = new_skys

        self.growing_status["counts"][visible_mask] += 1
        self.growing_status["grads"][visible_mask] += grads[
            visible_mask
        ]  # .clip(max=self.config.densify_grad_thresh * 3.0)
        # update the max screen size, as a ratio of number of pixels
        newradii = self.growing_status["radii"].detach()[visible_mask]
        self.growing_status["maxsize"][visible_mask] = torch.maximum(
            self.growing_status["maxsize"][visible_mask],
            newradii / float(min(self.growing_status["width"], self.growing_status["height"])),
        )

        for key in ["xys", "radii", "resp_times", "resp_visible", "resp_skys"]:
            if key in self.growing_status:
                del self.growing_status[key]

        torch.cuda.empty_cache()

    @torch.no_grad()
    def adjust_gaussians(
        self, training_callback_attributes: TrainingCallbackAttributes, do_densify=True, do_reset=False
    ):
        optimizers = training_callback_attributes.optimizers
        # Offset all the opacity reset logic by refine_every so that we don't
        # save checkpoints right when the opacity is reset (saves every 2k)
        # then cull
        # only split/cull if we've seen every image since opacity reset
        msg = f"original: {self.num_points}"
        if do_densify:
            # then we densify
            self.growing_status["grads"].nan_to_num_(0.0, 0.0, 0.0)
            avg_grad_norm = self.growing_status["grads"] / self.growing_status["counts"].clip(min=1)
            high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()

            """
            split gaissians
            """
            splits = (self.scales.exp().max(dim=-1).values > self.config.densify_scale_thresh).squeeze()
            splits &= high_grads
            if self.config.split_screen_thresh is not None:
                splits |= (self.growing_status["maxsize"] > self.config.split_screen_thresh).squeeze()
            if self.config.split_pixel_thresh is not None:
                splits |= (
                    self.growing_status["maxsize"] * min(self.growing_status["width"], self.growing_status["height"])
                    > self.config.split_pixel_thresh
                ).squeeze()
            # if self.config.use_scale_regularization:
            #     splits |= (
            #         self.scales.exp().amax(dim=-1) / self.scales.exp().amin(dim=-1)
            #         > self.config.scale_regularization_ratio
            #     ).squeeze()

            nsamps = self.config.n_split_samples
            split_params = self.split_gaussians(splits, nsamps)

            """
            duplicate gaussians
            """
            dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_scale_thresh).squeeze()
            dups &= high_grads
            # dups &= torch.logical_not(splits)

            # print(f"dup {dups.sum()} points")

            dup_params = self.dup_gaussians(dups)
            for name, param in self.gauss_params.items():
                self.gauss_params[name] = torch.nn.Parameter(
                    torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                )

            # append zeros to the growing status tensor
            for key in ["grads", "counts", "maxsize", "confs", "times", "skys"]:
                if key in self.growing_status:
                    self.growing_status[key] = torch.cat(
                        [
                            self.growing_status[key],
                            torch.zeros_like(split_params["scales"][:, 0], dtype=self.growing_status[key].dtype),
                            torch.zeros_like(dup_params["scales"][:, 0], dtype=self.growing_status[key].dtype),
                        ],
                        dim=0,
                    )

            split_idcs = torch.where(splits)[0]
            self.dup_in_all_optim(optimizers, split_idcs, nsamps)

            dup_idcs = torch.where(dups)[0]
            self.dup_in_all_optim(optimizers, dup_idcs, 1)

            # After a guassian is split into two new gaussians, the original one should also be pruned.
            splits_mask = torch.cat(
                (
                    splits,
                    torch.zeros(
                        nsamps * splits.sum() + dups.sum(),
                        device=self.device,
                        dtype=torch.bool,
                    ),
                )
            )
            msg += f", after split and dup: {self.num_points}"

            deleted_mask = self.cull_gaussians(splits_mask)
            msg += f", after cull: {self.num_points}"

        else:
            deleted_mask = None

        CONSOLE.print(msg)

        if deleted_mask is not None:
            self.remove_from_all_optim(optimizers, deleted_mask)

        if do_reset:
            # Reset value is set to be twice of the cull_alpha_thresh
            reset_value = self.config.cull_alpha_thresh * 2.0
            self.opacities.data = torch.clamp(
                self.opacities.data,
                max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
            )
            # reset the exp of optimizer
            optim = optimizers.optimizers["opacities"]
            param = optim.param_groups[0]["params"][0]
            param_state = optim.state[param]
            param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
            param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

        del self.growing_status["width"], self.growing_status["height"]

        if do_densify or do_reset:
            for key in ["grads", "counts", "maxsize", "confs", "times", "skys"]:
                if key in self.growing_status:
                    self.growing_status[key].fill_(0)

        torch.cuda.empty_cache()

    def after_train(self, training_callback_attributes: TrainingCallbackAttributes, step: int):

        gaussian_growth = (
            step >= self.start_refine_at
            and step < self.stop_refine_at
            and self.growth_stage in [GS.INITIAL, GS.JOINT, GS.FINETUNE]
        )
        confidence_update = self.config.consist_aware and self.growth_stage in [GS.INITIAL, GS.JOINT, GS.FINETUNE]

        if gaussian_growth or confidence_update:
            self.update_statis()

        if gaussian_growth:
            self.growing_status["iters"] += 1

            strategy_step = self.growing_status["iters"].item() - 1
            do_densify = strategy_step > 0 and strategy_step % self.config.refine_every == 0
            do_reset = False

            # if (
            #     self.config.enable_reset
            #     and strategy_step > 0
            #     and strategy_step % (self.config.refine_every * self.config.reset_alpha_every) == 0
            # ):
            #     do_reset = True
            #     do_densify = False

            if do_densify or do_reset:
                self.adjust_gaussians(training_callback_attributes, do_densify=do_densify, do_reset=do_reset)
                if do_reset:
                    self.after_reset(training_callback_attributes, indices=list(range(self.growth_index + 1)))

    # TODO:
    def after_reset(
        self, training_callback_attributes: TrainingCallbackAttributes, indices: List, masks: Optional[Dict] = None
    ):
        original_model_state = self.training
        original_param_state = self.selected_parameter
        original_train_idx = self.train_index
        self.set_parameter("all")
        self.train(mode=True)

        iters = self.config.joint_interval
        CONSOLE.print(f"after resetting, train with {iters} iters")

        for _ in tqdm.tqdm(range(iters)):

            self.train_index = random.choice(indices)

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

            background = "random"
            ret_visible = False
            gauss_confs = self._gauss_confs()
            query_values = None

            filter_sky = self.config.filter_sky and self.training

            outputs = self._train_view(
                index=self.train_index,
                background=background,
                ret_visible=ret_visible,
                query_values=query_values,
                gauss_confs=gauss_confs,
                filter_sky=filter_sky,
            )

            if masks is not None:
                outputs["mask"] = masks[self.train_index]

            loss_dict = self.get_loss_dict(outputs=outputs)
            if "mcmc_scale_reg" in loss_dict:
                del loss_dict["mcmc_scale_reg"]
            if "mcmc_opacity_reg" in loss_dict:
                del loss_dict["mcmc_opacity_reg"]

            training_callback_attributes.optimizers.zero_grad_all()
            loss = functools.reduce(torch.add, loss_dict.values())
            loss.backward()  # type: ignore

            for name, param in self.named_parameters():
                if param.requires_grad and getattr(param, "grad", None) is not None:
                    param.grad.nan_to_num_(0, 0, 0)
                    if "pose_adjustment" in name:
                        torch.nn.utils.clip_grad_norm_(param, max_norm=1e-3, norm_type=2)
            training_callback_attributes.optimizers.optimizer_step_all()

            self.update_statis()

        self.training = original_model_state
        self.selected_parameter = original_param_state
        self.train_index = original_train_idx
        self.set_parameter(original_param_state)
        self.train(mode=original_model_state)

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

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        self.growth_stage = GS.DONE
        self.growth_step = 0
        self.step = 1_000_000
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

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

    def get_sh_degree(self):
        if self.growth_stage == GS.INITIAL:
            return 0
        elif self.growth_stage == GS.CAMERA:
            return min(self.config.sh_degree, int(math.sqrt(self.growth_index - 1)))
        elif self.growth_stage in [GS.JOINT]:
            return min(self.config.sh_degree, int(math.sqrt(self.growth_index)))
        else:
            return self.config.sh_degree

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
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        if optional_c2w is not None:
            optimized_camera_to_world = optional_c2w.squeeze(0).unsqueeze(0)
        else:
            optimized_camera_to_world = camera.camera_to_worlds.squeeze(0).unsqueeze(0)

        # get the background color
        if background == "random":
            background = torch.rand(3, device=self.device)
        elif background == "black":
            background = torch.zeros(3, device=self.device)
        elif background == "white":
            background = torch.ones(3, device=self.device)
        else:
            raise NotImplementedError

        assert self.crop_box is None

        opacities_crop = self.opacities
        means_crop = self.means
        features_dc_crop = self.features_dc
        features_rest_crop = self.features_rest
        scales_crop = self.scales
        quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        screen_width = int(camera.width.item())
        screen_height = int(camera.height.item())

        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()

        render_mode = "RGB+ED"

        if self.config.sh_degree > 0:
            sh_degree_to_use = self.get_sh_degree()
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        if filter_sky:
            ret_visible = True
            if query_values is None:
                query_values = self.train_backgrounds[self.train_index].unsqueeze(0).unsqueeze(-1).type(torch.float32)
            else:
                query_values = torch.cat(
                    [
                        query_values,
                        self.train_backgrounds[self.train_index].unsqueeze(0).unsqueeze(-1).type(torch.float32),
                    ],
                    dim=-1,
                )

        render, alpha, render_distort, info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=int(camera.width.item()),
            height=int(camera.height.item()),
            packed=False,
            backgrounds=background.view(1, 3),
            near_plane=0.01,
            far_plane=1e5,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            distloss=True,
            rasterize_mode="antialiased",
            camera_model=(
                "equirectangular" if camera.camera_type.item() == CameraType.EQUIRECTANGULAR.value else "pinhole"
            ),
            ret_visible=ret_visible,
            query_values=query_values,
            gauss_confs=gauss_confs,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        if self.training:
            self.growing_status["xys"] = info["means2d"]  # [1, N, 2]
            self.growing_status["radii"] = info["radii"][0]  # [N]
            self.growing_status["width"] = screen_width
            self.growing_status["height"] = screen_height

        if filter_sky:
            self.growing_status["resp_times"] = info["accum_times"][0].detach()  # [N]
            self.growing_status["resp_visible"] = info["accum_visible"][0].detach()  # [N]
            self.growing_status["resp_skys"] = info["query_answers"][0].detach()[:, -1]  # [N]
            # remove sky from query answers
            if info["query_answers"].shape[-1] > 1:
                info["query_answers"] = info["query_answers"][:, :, :-1]
            else:
                info["accum_times"] = None
                info["accum_visible"] = None
                info["query_answers"] = None

        alpha = alpha[:, ...]
        rgb = render[:, ..., :3]
        if not self.training:
            rgb = torch.clamp(rgb, 0.0, 1.0).nan_to_num(0, 0, 0)

        depth_im = render[:, ..., 3:4]
        depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max())

        rets = {
            "undist_rgb": rgb.squeeze(0),
            "undist_depth": depth_im.squeeze(0),
            "render_distort": render_distort.squeeze(0),
            "background": background,
            "gaussians": self.num_points,
        }

        if info["accum_times"] is not None:
            rets["accum_times"] = info["accum_times"][0].detach()  # [N]
        if info["accum_visible"] is not None:
            rets["accum_visible"] = info["accum_visible"][0].detach()  # [N]
        if info["query_answers"] is not None:
            rets["query_answers"] = info["query_answers"][0].detach()  # [N, K]

        rets["radii"] = info["radii"][0].detach()  # [N, K]
        rets["rgb"] = rgb.squeeze(0)
        rets["depth"] = depth_im.squeeze(0)
        return rets
