from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from kornia.core.check import KORNIA_CHECK_SHAPE

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras, CameraType

from nerfstudio360.thirdparty import pytorch3d_transforms


E_TAN = 0.008
E_DEP = 0.05


def cvt_depth2radius(depths, cameras: Cameras):
    assert len(depths.shape) == 3 and depths.shape[-1] == 1
    rays = cameras.generate_rays(0, keep_shape=True)
    radius = depths * rays.metadata["directions_norm"]
    return radius


def cvt_radius2depth(radius, cameras: Cameras):
    assert len(radius.shape) == 3 and radius.shape[-1] == 1
    rays = cameras.generate_rays(0, keep_shape=True)
    depths = radius / rays.metadata["directions_norm"]
    return depths


def build_unposed_camera(ref_camera: Cameras):
    if len(ref_camera.camera_to_worlds.shape) == 2:
        ori_camera_to_worlds = torch.eye(4, dtype=torch.float32, device=ref_camera.camera_to_worlds.device)[:3, :]
    else:
        ori_camera_to_worlds = torch.eye(4, dtype=torch.float32, device=ref_camera.camera_to_worlds.device)[:3, :]
        ori_camera_to_worlds = (
            ori_camera_to_worlds.unsqueeze(0).repeat([ref_camera.camera_to_worlds.shape[0], 1, 1]).contiguous()
        )
    return Cameras(
        camera_to_worlds=ori_camera_to_worlds,
        width=ref_camera.width,
        height=ref_camera.height,
        fx=ref_camera.fx,
        fy=ref_camera.fy,
        cx=ref_camera.cx,
        cy=ref_camera.cy,
        distortion_params=ref_camera.distortion_params,
        camera_type=ref_camera.camera_type,
    )


def build_posed_camera(ref_camera: Cameras, novel_c2ws):
    return Cameras(
        camera_to_worlds=novel_c2ws.clone(),
        width=ref_camera.width,
        height=ref_camera.height,
        fx=ref_camera.fx,
        fy=ref_camera.fy,
        cx=ref_camera.cx,
        cy=ref_camera.cy,
        distortion_params=ref_camera.distortion_params,
        camera_type=ref_camera.camera_type,
    )


def cvt_persp2equir_camera(camera: Cameras):
    assert camera.camera_to_worlds.shape[0] == 1
    assert camera.camera_type.item() == CameraType.PERSPECTIVE.value
    H = max(int(camera.fx.item()), int(camera.fy.item())) * 4
    W = max(int(camera.fx.item()), int(camera.fy.item())) * 8
    device = camera.camera_to_worlds.device
    return Cameras(
        camera_to_worlds=camera.camera_to_worlds,
        width=torch.tensor([W]).to(device),
        height=torch.tensor([H]).to(device),
        fx=torch.tensor([H], dtype=torch.float32).to(device),
        fy=torch.tensor([H], dtype=torch.float32).to(device),
        cx=torch.tensor([W / 2.0], dtype=torch.float32).to(device),
        cy=torch.tensor([H / 2.0], dtype=torch.float32).to(device),
        distortion_params=None,
        camera_type=CameraType.EQUIRECTANGULAR.value,
    )


def get_grid_interp(image, xys, mode: str = "bilinear"):
    assert len(xys.shape) == 2 and xys.shape[-1] == 2 and xys.dtype == torch.float32
    assert image.shape[-1] in [1, 3] and len(image.shape) == 3
    h, w, c = image.shape
    n, _ = xys.shape
    image = image.permute(2, 0, 1).unsqueeze(0)
    xys = xys.view(1, 1, n, 2)
    ans = torch.nn.functional.grid_sample(image, grid=xys, mode=mode, align_corners=False)
    return ans.view(c, n).permute(1, 0)


def get_index_interp(image, xys):
    assert len(xys.shape) == 2 and xys.shape[-1] == 2 and xys.dtype in [torch.int32, torch.int64]
    assert image.shape[-1] in [1, 3] and len(image.shape) == 3
    h, w, c = image.shape
    index = xys[:, 0] + xys[:, 1] * w
    return torch.index_select(image.view(-1, c), index=index, dim=0)


@torch.no_grad()
def compute_consist_mask(
    this_camera: Cameras,
    this_c2w: torch.Tensor,
    this_depth: torch.Tensor,
    ref_camera: Cameras,
    ref_c2w: torch.Tensor,
    ref_depth: torch.Tensor,
    angle_check: bool = True,
    depth_check: bool = True,
):
    assert angle_check or depth_check
    this_camera = build_unposed_camera(this_camera)
    ref_camera = build_unposed_camera(ref_camera)

    c2w_R_im0 = this_c2w.squeeze()[:3, :3]  # [3, 3]
    c2w_T_im0 = this_c2w.squeeze()[:3, 3]  # [3]
    w2c_R_im0 = c2w_R_im0.T  # [3, 3]
    w2c_T_im0 = -torch.einsum("mn,n->m", c2w_R_im0.T, c2w_T_im0)  # [3]

    c2w_R_im1 = ref_c2w.squeeze()[:3, :3]  # [3, 3]
    c2w_T_im1 = ref_c2w.squeeze()[:3, 3]  # [3]
    w2c_R_im1 = c2w_R_im1.T  # [3, 3]
    w2c_T_im1 = -torch.einsum("mn,n->m", c2w_R_im1.T, c2w_T_im1)  # [3]

    _idt_rays = this_camera.generate_rays(0, keep_shape=True)
    points_im0 = this_depth * _idt_rays.directions * _idt_rays.metadata["directions_norm"] + _idt_rays.origins
    points_world = torch.einsum("hwn,mn->hwm", points_im0, c2w_R_im0) + c2w_T_im0.view(1, 1, 3)  # [n, 3]
    points_im01 = torch.einsum("hwn,mn->hwm", points_world, w2c_R_im1) + w2c_T_im1.view(1, 1, 3)  # [n, 3]

    _idt_rays = ref_camera.generate_rays(0, keep_shape=True)
    points_im1 = ref_depth * _idt_rays.directions * _idt_rays.metadata["directions_norm"] + _idt_rays.origins
    points_world = torch.einsum("hwn,mn->hwm", points_im1, c2w_R_im1) + c2w_T_im1.view(1, 1, 3)  # [n, 3]
    points_im10 = torch.einsum("hwn,mn->hwm", points_world, w2c_R_im0) + w2c_T_im0.view(1, 1, 3)  # [n, 3]

    if this_camera.camera_type.item() == CameraType.EQUIRECTANGULAR.value:
        polar = torch.atan2((points_im01[..., 0].square() + points_im01[..., 2].square()).sqrt(), points_im01[..., 1])
        azimuth = torch.atan2(points_im01[..., 0], -points_im01[..., 2])
        depth_im01 = points_im01.norm(dim=-1, keepdim=True).clone()
        interp_coords_im01 = torch.stack([azimuth / torch.pi, polar / torch.pi * 2 - 1], dim=-1)  # [h, w, 2]
        valid_mask_im01 = depth_im01.squeeze(-1) > 1e-3
    else:
        rud2rdf = torch.diag(torch.tensor([1, -1, -1], device=points_im01.device, dtype=points_im01.dtype))
        points_im01_rdf = torch.einsum("hwn,mn->hwm", points_im01, rud2rdf)
        intrinsic = this_camera.get_intrinsics_matrices().to(points_im01_rdf.device).squeeze()
        interp_coords_im01 = torch.einsum("hwn,mn->hwm", points_im01_rdf[..., :3], intrinsic.detach())
        depth_im01 = interp_coords_im01[..., 2:3].clone()
        valid_mask_z_im01 = interp_coords_im01[..., 2] > 1e-3
        interp_coords_im01 = interp_coords_im01[..., :2] / interp_coords_im01[..., 2:]
        width = this_camera.width.detach()
        height = this_camera.height.detach()
        shape = torch.stack([width, height], dim=-1).type(torch.float32).view(1, 1, 2)
        valid_mask_xy_im01 = torch.logical_and(interp_coords_im01 > 0, interp_coords_im01 < shape - 1)
        valid_mask_xy_im01 = torch.logical_and(valid_mask_xy_im01[..., 0], valid_mask_xy_im01[..., 1])
        interp_coords_im01 = interp_coords_im01 / shape * 2 - 1  # [h, w, 2]
        valid_mask_im01 = torch.logical_and(valid_mask_xy_im01, valid_mask_z_im01)

    consist = valid_mask_im01

    if angle_check:

        points_im0_warpback = torch.nn.functional.grid_sample(
            points_im10.unsqueeze(0).permute(0, 3, 1, 2),  # [1, 3, h, w]
            interp_coords_im01.unsqueeze(0),  # [1, h, w, 2]
            mode="bilinear",
            padding_mode="zeros",
        )  # [1, 3, h, w]

        points_im0_unit = F.normalize(points_im0, dim=-1, eps=1e-6)  # [h, w, 3]
        points_im0_warpback_unit = F.normalize(points_im0_warpback.squeeze(0).permute(1, 2, 0), dim=-1, eps=1e-6)

        up_im0 = 1 - (points_im0_unit * points_im0_warpback_unit).sum(dim=-1)
        down_im0 = 1 + (points_im0_unit * points_im0_warpback_unit).sum(dim=-1)
        dist_im0 = 2 * torch.sqrt(up_im0.clip(min=1e-12) / down_im0.clip(min=1e-12))

        angle_consist = dist_im0 < E_TAN
        consist = torch.logical_and(consist, angle_consist)

    if depth_check:
        if ref_camera.camera_type.item() == CameraType.EQUIRECTANGULAR.value:
            depth_im10 = points_im10.norm(dim=-1, keepdim=True).clone()
        else:
            rud2rdf = torch.diag(torch.tensor([1, -1, -1], device=points_im10.device, dtype=points_im10.dtype))
            points_im10_rdf = torch.einsum("hwn,mn->hwm", points_im10, rud2rdf)
            intrinsic = ref_camera.get_intrinsics_matrices().to(points_im10_rdf.device).squeeze()
            interp_coords_im10 = torch.einsum("hwn,mn->hwm", points_im10_rdf[..., :3], intrinsic.detach())
            depth_im10 = interp_coords_im10[..., 2:3].clone()

        depth_im0_warpback = torch.nn.functional.grid_sample(
            depth_im10.unsqueeze(0).permute(0, 3, 1, 2),  # [1, 1, h, w]
            interp_coords_im01.unsqueeze(0),  # [1, h, w, 2]
            mode="bilinear",
            padding_mode="zeros",
        )  # [1, 1, h, w]

        depth_consist = torch.logical_and(
            depth_im0_warpback.squeeze().clip(min=1e-3) / this_depth.squeeze().clip(min=1e-3) < 1 + E_DEP,
            this_depth.squeeze().clip(min=1e-3) / depth_im0_warpback.squeeze().clip(min=1e-3) < 1 + E_DEP,
        )
        consist = torch.logical_and(consist, depth_consist)

    return consist


@torch.no_grad()
def compute_interp_image(
    this_camera: Cameras,
    this_c2w: torch.Tensor,
    this_depth: torch.Tensor,
    ref_camera: Cameras,
    ref_c2w: torch.Tensor,
    ref_image: torch.Tensor,
):
    this_camera = build_unposed_camera(this_camera)
    ref_camera = build_unposed_camera(ref_camera)

    c2w_R_im0 = this_c2w.squeeze()[:3, :3]  # [3, 3]
    c2w_T_im0 = this_c2w.squeeze()[:3, 3]  # [3]

    c2w_R_im1 = ref_c2w.squeeze()[:3, :3]  # [3, 3]
    c2w_T_im1 = ref_c2w.squeeze()[:3, 3]  # [3]
    w2c_R_im1 = c2w_R_im1.T  # [3, 3]
    w2c_T_im1 = -torch.einsum("mn,n->m", c2w_R_im1.T, c2w_T_im1)  # [3]

    _idt_rays = this_camera.generate_rays(0, keep_shape=True)
    points_im0 = this_depth * _idt_rays.directions * _idt_rays.metadata["directions_norm"] + _idt_rays.origins
    points_world = torch.einsum("hwn,mn->hwm", points_im0, c2w_R_im0) + c2w_T_im0.view(1, 1, 3)  # [n, 3]
    points_im01 = torch.einsum("hwn,mn->hwm", points_world, w2c_R_im1) + w2c_T_im1.view(1, 1, 3)  # [n, 3]

    if this_camera.camera_type.item() == CameraType.EQUIRECTANGULAR.value:
        polar = torch.atan2((points_im01[..., 0].square() + points_im01[..., 2].square()).sqrt(), points_im01[..., 1])
        azimuth = torch.atan2(points_im01[..., 0], -points_im01[..., 2])
        depth_im01 = points_im01.norm(dim=-1, keepdim=True).clone()
        interp_coords_im01 = torch.stack([azimuth / torch.pi, polar / torch.pi * 2 - 1], dim=-1)  # [h, w, 2]
        valid_mask_im01 = depth_im01.squeeze(-1) > 1e-3
    else:
        rud2rdf = torch.diag(torch.tensor([1, -1, -1], device=points_im01.device, dtype=points_im01.dtype))
        points_im01_rdf = torch.einsum("hwn,mn->hwm", points_im01, rud2rdf)
        intrinsic = this_camera.get_intrinsics_matrices().to(points_im01_rdf.device).squeeze()
        interp_coords_im01 = torch.einsum("hwn,mn->hwm", points_im01_rdf[..., :3], intrinsic.detach())
        depth_im01 = interp_coords_im01[..., 2:3].clone()
        valid_mask_z_im01 = interp_coords_im01[..., 2] > 1e-3
        interp_coords_im01 = interp_coords_im01[..., :2] / interp_coords_im01[..., 2:]
        width = this_camera.width.detach()
        height = this_camera.height.detach()
        shape = torch.stack([width, height], dim=-1).type(torch.float32).view(1, 1, 2)
        valid_mask_xy_im01 = torch.logical_and(interp_coords_im01 > 0, interp_coords_im01 < shape - 1)
        valid_mask_xy_im01 = torch.logical_and(valid_mask_xy_im01[..., 0], valid_mask_xy_im01[..., 1])
        interp_coords_im01 = interp_coords_im01 / shape * 2 - 1  # [h, w, 2]
        valid_mask_im01 = torch.logical_and(valid_mask_xy_im01, valid_mask_z_im01)

    image_warpback = torch.nn.functional.grid_sample(
        ref_image.unsqueeze(0).permute(0, 3, 1, 2),  # [1, 3, h, w]
        interp_coords_im01.unsqueeze(0),  # [1, h, w, 2]
        mode="bilinear",
        padding_mode="zeros",
    )  # [1, 3, h, w]

    image_warpback = image_warpback.squeeze(0).permute(1, 2, 0) * valid_mask_im01.unsqueeze(-1).type(torch.float32)

    return image_warpback, valid_mask_im01
