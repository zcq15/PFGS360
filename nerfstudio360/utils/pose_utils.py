import copy
import math
import kornia
import ipdb
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio360.utils import camera_utils
from nerfstudio360.utils.camera_utils import E_TAN, E_DEP
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from nerfstudio.utils.rich_utils import CONSOLE
import time
import cv2


@torch.no_grad()
def _match_with_opencv(src_image: torch.Tensor, tar_image: torch.Tensor):

    device = src_image.device
    src_image = src_image.permute(2, 0, 1).unsqueeze(0)
    tar_image = tar_image.permute(2, 0, 1).unsqueeze(0)

    src_image = (src_image.squeeze(0).permute(1, 2, 0) * 255).clip(min=0, max=255).byte().cpu().numpy()
    src_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2BGR)

    tar_image = (tar_image.squeeze(0).permute(1, 2, 0) * 255).clip(min=0, max=255).byte().cpu().numpy()
    tar_image = cv2.cvtColor(tar_image, cv2.COLOR_RGB2BGR)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src_image, None)
    kp2, des2 = sift.detectAndCompute(tar_image, None)
    ratio = 0.85
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m1, m2 in raw_matches:
        if m1.distance < ratio * m2.distance:
            good_matches.append([m1])

    matches_im0 = []
    matches_im1 = []
    for gm in good_matches:
        matches_im0.append(kp1[gm[0].queryIdx].pt)
        matches_im1.append(kp2[gm[0].trainIdx].pt)

    matches_im0 = np.array(matches_im0)
    matches_im1 = np.array(matches_im1)

    matches_im0 = torch.from_numpy(matches_im0).type(torch.int64).to(device)
    matches_im1 = torch.from_numpy(matches_im1).type(torch.int64).to(device)

    _, index, counts = torch.unique(matches_im0, return_inverse=True, return_counts=True, dim=0)
    valid_im0 = torch.index_select(counts, index=index, dim=0) == 1
    _, index, counts = torch.unique(matches_im1, return_inverse=True, return_counts=True, dim=0)
    valid_im1 = torch.index_select(counts, index=index, dim=0) == 1

    valid = torch.logical_and(valid_im0, valid_im1)
    matches_im0 = matches_im0[valid]
    matches_im1 = matches_im1[valid]

    return matches_im0, matches_im1


@torch.no_grad()
def _get_match_pairs(
    src_image: torch.Tensor, src_valid: torch.Tensor, tar_image: torch.Tensor, tar_valid: torch.Tensor
):

    assert KORNIA_CHECK_SHAPE(src_image, ["H", "W", "3"])
    assert KORNIA_CHECK_SHAPE(src_valid, ["H", "W"])
    assert KORNIA_CHECK_SHAPE(tar_image, ["H", "W", "3"])
    assert KORNIA_CHECK_SHAPE(tar_valid, ["H", "W"])

    matches_im0, matches_im1 = _match_with_opencv(src_image, tar_image)

    # valid_im0 = camera_utils.get_index_interp(src_valid.unsqueeze(-1).type(torch.float32), matches_im0).squeeze(-1)
    # valid_im1 = camera_utils.get_index_interp(tar_valid.unsqueeze(-1).type(torch.float32), matches_im1).squeeze(-1)

    # matches_im0 = matches_im0[torch.logical_and(valid_im0, valid_im1)].clone().detach()
    # matches_im1 = matches_im1[torch.logical_and(valid_im0, valid_im1)].clone().detach()

    return matches_im0, matches_im1


@torch.no_grad()
def normalize_E_by_unit_t(E):
    """
    E: [t, 3, 3]
    return: E_normalized [t, 3, 3]
    """
    U, _, Vh = torch.linalg.svd(E)

    S = torch.zeros(E.shape[0], 3, device=E.device)
    S[:, 0] = 1.0
    S[:, 1] = 1.0

    E = U @ torch.diag_embed(S) @ Vh
    return E


@torch.no_grad()
def estimate_E_batched_svd(x0, x1):
    """
    x0, x1: [t, p, 3] unit bearing vectors
    return: E [t, 3, 3]
    """
    x0x, x0y, x0z = x0.unbind(dim=-1)
    x1x, x1y, x1z = x1.unbind(dim=-1)

    A = torch.stack(
        [
            x1x * x0x,
            x1x * x0y,
            x1x * x0z,
            x1y * x0x,
            x1y * x0y,
            x1y * x0z,
            x1z * x0x,
            x1z * x0y,
            x1z * x0z,
        ],
        dim=-1,
    )  # [t, p, 9]

    _, _, Vh = torch.linalg.svd(A)
    E = Vh[:, -1].reshape(-1, 3, 3)

    # Enforce essential constraint
    U, _, Vh = torch.linalg.svd(E)
    S = torch.zeros(E.shape[0], 3, device=E.device)
    S[:, :2] = 1.0

    E = U @ torch.diag_embed(S) @ Vh
    return normalize_E_by_unit_t(E)


def _filter_match_pairs(
    src_unposed_cameras: Cameras,
    tar_unposed_cameras: Cameras,
    matches_im0: torch.Tensor,
    matches_im1: torch.Tensor,
    config: CameraOptimizerConfig,
):
    assert len(src_unposed_cameras) == 1
    assert len(tar_unposed_cameras) == 1

    _idt_rays = src_unposed_cameras.generate_rays(0, keep_shape=True)
    points_im0_unit = camera_utils.get_index_interp(_idt_rays.directions, matches_im0)  # [n, 3]
    _idt_rays = tar_unposed_cameras.generate_rays(0, keep_shape=True)
    points_im1_unit = camera_utils.get_index_interp(_idt_rays.directions, matches_im1)  # [n, 3]

    ransac_thresh = E_TAN
    ransac_lr = 0.1
    ransac_iters = 500
    ransac_epochs = 1000
    ransac_samples = 8

    camera_optimizer_ransac = config.setup(num_cameras=ransac_epochs, device="cuda")
    state_dict = camera_optimizer_ransac.state_dict()
    state_dict["pose_adjustment"][:, :3].fill_(1e-3)
    state_dict["pose_adjustment"] = state_dict["pose_adjustment"].clone().contiguous()
    camera_optimizer_ransac.load_state_dict(state_dict)

    points_im0_unit_ransac = []
    points_im1_unit_ransac = []
    for _ in range(ransac_epochs):
        selected_index = np.random.choice(matches_im0.shape[0], ransac_samples, replace=False)
        points_im0_unit_ransac.append(points_im0_unit[selected_index])
        points_im1_unit_ransac.append(points_im1_unit[selected_index])
    points_im0_unit_ransac = torch.stack(points_im0_unit_ransac, dim=0)  # [t, p, 3]
    points_im1_unit_ransac = torch.stack(points_im1_unit_ransac, dim=0)  # [t, p, 3]

    # selected_index = np.random.choice(matches_im0.shape[0], ransac_samples * ransac_epochs, replace=True)
    # points_im0_unit_ransac = points_im0_unit[selected_index].reshape(ransac_epochs, ransac_samples, 3)
    # points_im1_unit_ransac = points_im1_unit[selected_index].reshape(ransac_epochs, ransac_samples, 3)

    E = estimate_E_batched_svd(points_im0_unit_ransac, points_im1_unit_ransac)

    with torch.no_grad():
        Ex0 = torch.einsum("tmn,pn->tpm", E, points_im0_unit)  # [t, p, 3]
        x1Ex0 = torch.einsum("pn,tpn->tp", points_im1_unit, Ex0)  # [t, p]
        distance = x1Ex0.abs() / (
            Ex0.norm(dim=-1).clip(min=1e-6) * points_im1_unit.unsqueeze(0).norm(dim=-1).clip(min=1e-6)
        )

        hits = (distance < ransac_thresh).type(torch.int32).sum(dim=-1)  # [t]
        max_hit = hits.max()
        if (hits == max_hit).sum().item() == 1:
            max_hit_index = hits.argmax().item()
        else:
            all_index = torch.arange(hits.shape[0]).to(hits.device).view(-1)[hits == max_hit]
            dist_list = distance.sum(dim=-1)[hits == max_hit]
            max_hit_index = all_index[dist_list.argmin().item()].item()

        distance = distance[max_hit_index]
        inlier = distance < ransac_thresh
        matches_im0 = matches_im0[inlier]
        matches_im1 = matches_im1[inlier]

    return matches_im0, matches_im1


def _get_pnp_pose(
    world_points: torch.Tensor,
    camera_dirs: torch.Tensor,
    match_confs: torch.Tensor,
    initial_poses: torch.Tensor,
    config: CameraOptimizerConfig,
):
    assert KORNIA_CHECK_SHAPE(world_points, ["N", "3"])
    assert KORNIA_CHECK_SHAPE(camera_dirs, ["N", "3"])
    assert KORNIA_CHECK_SHAPE(match_confs, ["N"])
    assert KORNIA_CHECK_SHAPE(initial_poses, ["1", "6"])

    ransac_thresh = E_TAN
    ransac_lr = 0.1
    ransac_iters = 500
    ransac_epochs = 1000
    ransac_samples = 8

    camera_optimizer_ransac = config.setup(num_cameras=ransac_epochs, device="cuda")
    state_dict = camera_optimizer_ransac.state_dict()
    state_dict["pose_adjustment"] = initial_poses.expand(ransac_epochs, 6).clone().detach()
    state_dict["pose_adjustment"] = state_dict["pose_adjustment"].clone().contiguous()
    camera_optimizer_ransac.load_state_dict(state_dict)

    world_points_ransac = []
    camera_dirs_ransac = []
    for _ in range(ransac_epochs):
        selected_index = np.random.choice(world_points.shape[0], ransac_samples, replace=False)
        world_points_ransac.append(world_points[selected_index])
        camera_dirs_ransac.append(camera_dirs[selected_index])
    world_points_ransac = torch.stack(world_points_ransac, dim=0)  # [t, p, 3]
    camera_dirs_ransac = torch.stack(camera_dirs_ransac, dim=0)  # [t, p, 3]

    # selected_index = np.random.choice(world_points.shape[0], ransac_samples * ransac_epochs, replace=True)
    # world_points_ransac = world_points[selected_index].reshape(ransac_epochs, ransac_samples, 3)
    # camera_dirs_ransac = camera_dirs[selected_index].reshape(ransac_epochs, ransac_samples, 3)

    optimizer = torch.optim.Adam(
        [
            {"params": camera_optimizer_ransac.parameters(), "lr": ransac_lr},
        ],
        lr=ransac_lr,
        eps=1e-15,
    )
    for iter in range(ransac_iters):
        for param_group in optimizer.param_groups:
            param_group["lr"] = 1e-4 + (ransac_lr - 1e-4) * max(0.5 * (math.cos(iter / ransac_iters * math.pi) + 1), 0)
        optimizer.zero_grad()
        c2w_im1 = camera_optimizer_ransac.all_poses()  # update current camera
        c2w_R_im1 = c2w_im1[:, :3, :3]  # [t, 3, 3]
        c2w_T_im1 = c2w_im1[:, :3, 3]  # [t, 3]
        inv_c2w_R_im1 = c2w_R_im1.permute(0, 2, 1)  # R^T, [t, 3, 3]
        inv_c2w_T_im1 = -torch.einsum("tmn,tn->tm", inv_c2w_R_im1, c2w_T_im1)  # -R^T@t, [t, 3]

        w2c_points_ransac = torch.einsum("tmn,tkn->tkm", inv_c2w_R_im1, world_points_ransac) + inv_c2w_T_im1.view(
            -1, 1, 3
        )  # [t, p ,3]
        w2c_points_ransac = w2c_points_ransac / w2c_points_ransac.norm(dim=-1, keepdim=True).clip(min=1e-3)
        up = 1 - (w2c_points_ransac * camera_dirs_ransac).sum(dim=-1)  # [t, p]
        down = 1 + (w2c_points_ransac * camera_dirs_ransac).sum(dim=-1)  # [t, p]
        loss = (2 * torch.sqrt(up.clip(min=1e-12) / down.clip(min=1e-12))).mean(dim=-1).sum()  # [t, p] -> [t] -> [1]

        loss.backward()
        camera_optimizer_ransac.pose_adjustment.grad.nan_to_num_(0, 0, 0)
        torch.nn.utils.clip_grad_value_(camera_optimizer_ransac.pose_adjustment, clip_value=1e-2)
        optimizer.step()

    with torch.no_grad():
        c2w_im1 = camera_optimizer_ransac.all_poses()  # update current camera
        c2w_R_im1 = c2w_im1[:, :3, :3]  # [t, 3, 3]
        c2w_T_im1 = c2w_im1[:, :3, 3]  # [t, 3]
        inv_c2w_R_im1 = c2w_R_im1.permute(0, 2, 1)  # R^T
        inv_c2w_T_im1 = -torch.einsum("tmn,tn->tm", inv_c2w_R_im1, c2w_T_im1)  # -R^T@t

        points_im01 = torch.einsum("tmn,kn->tkm", inv_c2w_R_im1, world_points) + inv_c2w_T_im1.view(
            -1, 1, 3
        )  # [t, n, 3]
        points_im01_unit = points_im01 / points_im01.norm(dim=-1, keepdim=True).clip(min=1e-3)

        up = 1 - (points_im01_unit * camera_dirs).sum(dim=-1)
        down = 1 + (points_im01_unit * camera_dirs).sum(dim=-1)
        distance = 2 * torch.sqrt(up.clip(min=1e-12) / down.clip(min=1e-12))  # [t, n]

        confs = (distance < ransac_thresh).type(torch.float32) * match_confs.view(1, -1)
        confs = confs.sum(dim=-1)  # [t]
        max_hit_index = confs.argmax().item()

        # hits = (distance < ransac_thresh).type(torch.int32).sum(dim=-1)  # [t]
        # max_hit = hits.max()
        # if (hits == max_hit).sum().item() == 1:
        #     max_hit_index = hits.argmax().item()
        # else:
        #     all_index = torch.arange(hits.shape[0]).to(hits.device).view(-1)[hits == max_hit]
        #     dist_list = distance.sum(dim=-1)[hits == max_hit]
        #     max_hit_index = all_index[dist_list.argmin().item()].item()

        pose_adjustment = camera_optimizer_ransac.pose_adjustment.data.clone().detach()
        poses = pose_adjustment[max_hit_index : max_hit_index + 1].clone().detach()

        distance = distance[max_hit_index]
        inlier = distance < ransac_thresh
        world_points = world_points[inlier]
        camera_dirs = camera_dirs[inlier]
        match_confs = match_confs[inlier]

        return poses, world_points, camera_dirs, match_confs


def _refine_pnp_pose(
    world_points: torch.Tensor, camera_dirs: torch.Tensor, coarse_poses: torch.Tensor, config: CameraOptimizerConfig
):
    assert KORNIA_CHECK_SHAPE(world_points, ["N", "3"])
    assert KORNIA_CHECK_SHAPE(camera_dirs, ["N", "3"])
    assert KORNIA_CHECK_SHAPE(coarse_poses, ["1", "6"])

    refine_lr = 0.1
    refine_iters = 500
    camera_optimizer = config.setup(num_cameras=1, device="cuda")
    state_dict = camera_optimizer.state_dict()
    state_dict["pose_adjustment"] = coarse_poses.clone().detach()
    state_dict["pose_adjustment"] = state_dict["pose_adjustment"].clone().contiguous()
    camera_optimizer.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(
        [
            {"params": camera_optimizer.parameters(), "lr": refine_lr},
        ],
        lr=refine_lr,
        eps=1e-15,
    )
    for iter in range(refine_iters):
        for param_group in optimizer.param_groups:
            param_group["lr"] = 1e-4 + (refine_lr - 1e-4) * max(0.5 * (math.cos(iter / refine_iters * math.pi) + 1), 0)
        optimizer.zero_grad()
        c2w_im1 = camera_optimizer.all_poses().squeeze(0)  # update current camera
        c2w_R_im1 = c2w_im1[:3, :3]  # [3, 3]
        c2w_T_im1 = c2w_im1[:3, 3]  # [3]
        inv_c2w_R_im1 = c2w_R_im1.T  # R^T, [3, 3]
        inv_c2w_T_im1 = -torch.einsum("mn,n->m", inv_c2w_R_im1, c2w_T_im1)  # -R^T@t, [3]

        w2c_points = torch.einsum("mn,kn->km", inv_c2w_R_im1, world_points) + inv_c2w_T_im1.view(1, 3)  # [p ,3]
        w2c_points = w2c_points / w2c_points.norm(dim=-1, keepdim=True).clip(min=1e-3)
        up = 1 - (w2c_points * camera_dirs).mean()  # [1]
        down = 1 + (w2c_points * camera_dirs).mean()  # [1]
        loss = 2 * torch.sqrt(up.clip(min=1e-12) / down.clip(min=1e-12))  # [1]

        loss.backward()
        camera_optimizer.pose_adjustment.grad.nan_to_num_(0, 0, 0)
        torch.nn.utils.clip_grad_value_(camera_optimizer.pose_adjustment, clip_value=1e-3)
        optimizer.step()

    state_dict = camera_optimizer.state_dict()
    poses = state_dict["pose_adjustment"].clone().detach()

    return poses


def compute_confidence(
    matches_im0,  # xys
    matches_im1,  # xys
    src_valid: torch.Tensor,
    tar_valid: torch.Tensor,
    spherical: bool = True,
):
    assert KORNIA_CHECK_SHAPE(src_valid, ["H", "W"])
    assert KORNIA_CHECK_SHAPE(tar_valid, ["H", "W"])
    valid_im0 = camera_utils.get_index_interp(src_valid.unsqueeze(-1).type(torch.float32), matches_im0).squeeze(-1)
    valid_im1 = camera_utils.get_index_interp(tar_valid.unsqueeze(-1).type(torch.float32), matches_im1).squeeze(-1)
    valid_im0 = valid_im0.type(torch.float32)
    valid_im1 = valid_im1.type(torch.float32)
    if spherical:
        conf_im0 = torch.sin((matches_im0[:, 1].type(torch.float32) + 0.5) / src_valid.shape[0] * torch.pi).clip(min=0)
        conf_im1 = torch.sin((matches_im1[:, 1].type(torch.float32) + 0.5) / tar_valid.shape[0] * torch.pi).clip(min=0)
    else:
        conf_im0 = 1.0
        conf_im1 = 1.0
    return valid_im0 * valid_im1 * conf_im0 * conf_im1


def compute_camera_pose(
    src_cameras: Cameras,
    src_uids: list,
    src_poses: torch.Tensor,
    src_images: torch.Tensor,
    src_depths: torch.Tensor,
    src_valids: torch.Tensor,
    tar_cameras: Cameras,
    tar_uids: list,
    tar_images: torch.Tensor,
    tar_valids: torch.Tensor,
    config: CameraOptimizerConfig,
    spherical: bool = True,
):
    assert len(src_cameras) == len(src_uids)
    assert len(tar_cameras) == len(tar_uids)
    assert len(src_poses) == len(src_uids)
    assert len(tar_uids) == 1
    assert KORNIA_CHECK_SHAPE(src_poses, ["N", "6"])
    assert KORNIA_CHECK_SHAPE(src_images, ["N", "H", "W", "3"])
    assert KORNIA_CHECK_SHAPE(src_depths, ["N", "H", "W", "1"])
    assert KORNIA_CHECK_SHAPE(src_valids, ["N", "H", "W"])
    assert KORNIA_CHECK_SHAPE(tar_images, ["1", "H", "W", "3"])
    assert KORNIA_CHECK_SHAPE(tar_valids, ["1", "H", "W"])

    src_unposed_cameras = camera_utils.build_unposed_camera(src_cameras)
    tar_unposed_cameras = camera_utils.build_unposed_camera(tar_cameras)
    src_pose_optimizer = config.setup(num_cameras=len(src_cameras), device="cuda")
    src_pose_optimizer.update_poses(src_poses, selected_indices=list(range(len(src_cameras))))

    world_points = []
    camera_dirs = []
    camera_uids = []
    inliers_num = []
    match_confs = []

    for idx in range(len(src_cameras)):
        start_time = time.perf_counter()
        matches_im0, matches_im1 = _get_match_pairs(src_images[idx], src_valids[idx], tar_images[0], tar_valids[0])
        end_time = time.perf_counter()
        assert matches_im0.shape[0] == matches_im1.shape[0]
        CONSOLE.print(
            f"sift match pairs for view {src_uids[idx]} and {tar_uids[0]}: {matches_im0.shape[0]} pairs, {end_time - start_time:.3f} seconds"
        )
        if matches_im0.shape[0] < 100 and len(src_cameras) > 1:
            continue
        start_time = time.perf_counter()
        matches_im0, matches_im1 = _filter_match_pairs(
            src_unposed_cameras[idx : idx + 1],
            tar_unposed_cameras,
            matches_im0,
            matches_im1,
            config,
        )
        end_time = time.perf_counter()
        CONSOLE.print(
            f"filter epipolar match pairs for view {src_uids[idx]} and {tar_uids[0]}: {matches_im0.shape[0]} pairs, {end_time - start_time:.3f} seconds"
        )

        start_time = time.perf_counter()
        confs = compute_confidence(matches_im0, matches_im1, src_valids[idx], tar_valids[0], spherical=spherical)
        end_time = time.perf_counter()
        CONSOLE.print(
            f"compute confidence time for view {src_uids[idx]} and {tar_uids[0]}: {end_time - start_time:.3f} seconds"
        )

        assert matches_im0.shape[0] == matches_im1.shape[0]
        if matches_im0.shape[0] < 100 and len(src_cameras) > 1:
            continue

        with torch.no_grad():
            camera_uids.append(src_uids[idx])
            match_confs.append(confs)

            _posed_camera = camera_utils.build_posed_camera(
                src_unposed_cameras[idx : idx + 1], src_pose_optimizer.selected_poses([idx])
            )
            _posed_rays = _posed_camera.generate_rays(0, keep_shape=True)
            _posed_dirs = camera_utils.get_index_interp(
                _posed_rays.directions * _posed_rays.metadata["directions_norm"], matches_im0
            )
            _posed_origins = camera_utils.get_index_interp(_posed_rays.origins, matches_im0)
            _posed_depths = camera_utils.get_index_interp(src_depths[idx], matches_im0)
            _points_im0 = _posed_origins + _posed_dirs * _posed_depths  # [n, 3]
            _unposed_rays = tar_unposed_cameras.generate_rays(0, keep_shape=True)
            _unposed_unitdirs = camera_utils.get_index_interp(_unposed_rays.directions, matches_im1)

            world_points.append(_points_im0)
            camera_dirs.append(_unposed_unitdirs)
            inliers_num.append(matches_im0.shape[0])

    match_confs = torch.cat(match_confs, dim=0).clone().detach()
    world_points = torch.cat(world_points, dim=0).clone().detach()
    camera_dirs = torch.cat(camera_dirs, dim=0).clone().detach()
    inliers_num = torch.tensor(inliers_num)

    initial_index = inliers_num.argmax().item()
    initial_poses = src_poses[initial_index : initial_index + 1].clone().detach()

    start_time = time.perf_counter()
    coarse_poses, world_points, camera_dirs, match_confs = _get_pnp_pose(
        world_points, camera_dirs, match_confs, initial_poses, config
    )
    end_time = time.perf_counter()
    CONSOLE.print(
        f"final match pairs for views {camera_uids} and {tar_uids[0]}: {world_points.shape[0]} pairs, {end_time - start_time:.3f} seconds"
    )

    return coarse_poses
    # tar_poses = _refine_pnp_pose(world_points, camera_dirs, coarse_poses, config)
    # return tar_poses
