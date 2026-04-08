import copy
import json
import os
from enum import Enum

import matplotlib.pyplot as plt
import torch
from evo.core.trajectory import PosePath3D
from evo.tools import plot

from nerfstudio360.thirdparty.cf3dgs_camera_alignment import align_cameras_and_worlds

json.encoder.FLOAT_REPR = lambda o: format(o, ".4f")


class GrowthState(Enum):
    INITIAL = 0
    CAMERA = 1
    JOINT = 2
    FINETUNE = 3
    EVALUATE = 4
    DONE = 5


def selected_poses(self, selected_indices) -> torch.Tensor:
    """Apply the pose correction to the world-to-camera matrix in a Camera object"""
    assert not self.config.mode == "off"
    if isinstance(selected_indices, int):
        return self(torch.tensor([selected_indices], dtype=torch.long)).to(self.pose_adjustment.device).squeeze(0)  # type: ignore
    elif isinstance(selected_indices, list):
        return self(torch.tensor(selected_indices, dtype=torch.long)).to(self.pose_adjustment.device)  # type: ignore
    else:
        raise ValueError


def all_poses(self) -> torch.Tensor:
    """Apply the pose correction to the world-to-camera matrix in a Camera object"""
    assert not self.config.mode == "off"
    indices = list(range(self.num_cameras))
    return self(torch.tensor(indices, dtype=torch.long)).to(self.pose_adjustment.device)  # type: ignore


def update_poses(self, new_poses: torch.Tensor, selected_indices: list) -> None:
    assert len(new_poses) == len(selected_indices)
    state_dict = copy.deepcopy(self.state_dict())
    for i in range(len(selected_indices)):
        index = selected_indices[i]
        pose = new_poses[i].clone().detach()
        state_dict["pose_adjustment"][index] = pose.to(state_dict["pose_adjustment"].device)
    state_dict["pose_adjustment"] = state_dict["pose_adjustment"].contiguous()
    self.load_state_dict(state_dict)


def plot_camera_pose(ref_poses, est_poses, save_dir, filename=None, vid=False, elev=60, azim=60):
    ref_poses = [pose for pose in ref_poses]
    if isinstance(est_poses, dict):
        est_poses = [pose for k, pose in est_poses.items()]
    else:
        est_poses = [pose for pose in est_poses]
    traj_ref = PosePath3D(poses_se3=ref_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False)
    if vid:
        for p_idx in range(len(ref_poses)):
            fig = plt.figure()
            current_est_aligned = traj_est_aligned.poses_se3[: p_idx + 1]
            current_ref = traj_ref.poses_se3[: p_idx + 1]
            current_est_aligned = PosePath3D(poses_se3=current_est_aligned)
            current_ref = PosePath3D(poses_se3=current_ref)
            traj_by_label = {
                # "estimate (not aligned)": traj_est,
                "Pred (aligned)": current_est_aligned,
                "Ground-truth": current_ref,
            }
            plot_mode = plot.PlotMode.xyz
            # ax = plot.prepare_axis(fig, plot_mode, 111)
            ax = fig.add_subplot(111, projection="3d")
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.zaxis.set_tick_params(labelleft=False)
            colors = ["r", "b"]
            styles = ["-", "--"]

            for idx, (label, traj) in enumerate(traj_by_label.items()):
                plot.traj(ax, plot_mode, traj, styles[idx], colors[idx], label)
                # break
            # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
            ax.view_init(elev=elev, azim=azim)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            plt.tight_layout()
            os.makedirs(os.path.join(save_dir, "pose_vid"), exist_ok=True)
            pose_vis_path = os.path.join(save_dir, "pose_vid", "pose_{:03d}.png".format(p_idx))
            print(pose_vis_path)
            fig.savefig(pose_vis_path, dpi=600)

    # else:

    fig = plt.figure()
    traj_by_label = {
        # "estimate (not aligned)": traj_est,
        "Pred (aligned)": traj_est_aligned,
        "Ground-truth": traj_ref,
    }
    plot_mode = plot.PlotMode.xyz
    # ax = plot.prepare_axis(fig, plot_mode, 111)
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.zaxis.set_tick_params(labelleft=False)
    colors = ["r", "b"]
    styles = ["-", "--"]

    for idx, (label, traj) in enumerate(traj_by_label.items()):
        plot.traj(ax, plot_mode, traj, styles[idx], colors[idx], label)
        # break
    # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    ax.view_init(elev=elev, azim=azim)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    plt.tight_layout()
    pose_vis_path = os.path.join(save_dir, "pose.png" if filename is None else filename)
    fig.savefig(pose_vis_path, dpi=600)


@torch.no_grad()
def align_cameras(self):
    """
    transform camera poses in the model to real world camera poses and get the errors
    """
    c2ws_gt = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat([self.num_train_data, 1, 1])
    c2ws_pred = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat([self.num_train_data, 1, 1])
    c2ws_gt[:, :3, :] = self.groundtruth_cameras.camera_to_worlds.clone().detach()
    c2ws_pred[:, :3, :] = self.camera_optimizer.all_poses()[:, :3].clone().detach()

    R_edit = torch.diag(torch.tensor([1, -1, -1], device=c2ws_gt.device, dtype=c2ws_gt.dtype))
    c2ws_gt[:, :3, :3] = c2ws_gt[:, :3, :3] @ R_edit
    c2ws_pred[:, :3, :3] = c2ws_pred[:, :3, :3] @ R_edit

    rpe_t, rpe_r, ate, aligned_c2ws_gt, aligned_c2ws_pred = align_cameras_and_worlds(c2ws_gt, c2ws_pred)
    self.register_buffer("RPE_t", rpe_t.view([1]).clone())
    self.register_buffer("RPE_r", rpe_r.view([1]).clone())
    self.register_buffer("ATE", ate.view([1]).clone())
