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

"""
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import copy
import os.path
import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import ipdb
import torch
import torch.distributed as dist
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE


from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig, Pipeline
from PIL import Image
import numpy as np
import open3d
import cv2

from ipdb import set_trace


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


@dataclass
class ColmapFreePipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ColmapFreePipeline)
    """target class to instantiate"""
    # datamanager: DataManagerConfig = field(default_factory=lambda: DataManagerConfig())
    # """specifies the datamanager config"""
    # model: ModelConfig = field(default_factory=lambda: ModelConfig())
    # """specifies the model config"""
    suffix: Optional[str] = None
    """suffix to config model dir"""


class ColmapFreePipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    config: ColmapFreePipelineConfig
    datamanager: VanillaDataManager

    def __init__(
        self,
        config: ColmapFreePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__()
        self.config = config
        self.suffix = config.suffix
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        def _load_depth_images(depth_filenames, depth_unit_scale_factor):
            if depth_filenames is not None:
                depth_images = []
                for fn in depth_filenames:
                    depth = cv2.imread(str(fn), cv2.IMREAD_ANYDEPTH).astype(np.float32) * depth_unit_scale_factor
                    depth_images.append(torch.from_numpy(depth))
                depth_images = torch.stack(depth_images, dim=0).unsqueeze(-1)
            else:
                depth_images = None
            return depth_images

        def _load_backgound_masks(backgound_filenames):
            if backgound_filenames is not None:
                backgound_masks = []
                for fn in backgound_filenames:
                    backgound_mask = cv2.imread(str(fn), cv2.IMREAD_ANYDEPTH)
                    backgound_masks.append(torch.from_numpy(backgound_mask) > 0)
                backgound_masks = torch.stack(backgound_masks, dim=0)
            else:
                backgound_masks = None
            return backgound_masks

        def _composite_images_with_background(data, background=torch.tensor([0, 0, 0], dtype=torch.float32)):
            composited_images = []
            for item in data:
                image_raw = item["image"]
                if image_raw.shape[2] == 4:
                    alpha = image_raw[..., -1].unsqueeze(-1).repeat((1, 1, 3))
                    composited_images.append(alpha * image_raw[..., :3] + (1 - alpha) * background.to(image_raw.device))
                else:
                    composited_images.append(image_raw.clone())
            return torch.stack(composited_images, dim=0)

        def _load_cameras(cameras):
            return copy.deepcopy(cameras)

        # if "depth_filenames" in self.datamanager.train_dataset.metadata:
        #     train_depths = _load_depth_images(
        #         self.datamanager.train_dataset.metadata["depth_filenames"],
        #         self.datamanager.train_dataset.metadata["depth_unit_scale_factor"],
        #     )
        # else:
        #     train_depths = None

        # if "depth_filenames" in self.datamanager.eval_dataset.metadata:
        #     eval_depths = _load_depth_images(
        #         self.datamanager.eval_dataset.metadata["depth_filenames"],
        #         self.datamanager.eval_dataset.metadata["depth_unit_scale_factor"],
        #     )
        # else:
        #     eval_depths = None

        train_images = _composite_images_with_background(self.datamanager.cached_train)
        train_cameras = _load_cameras(self.datamanager.train_dataset.cameras)
        eval_images = _composite_images_with_background(self.datamanager.cached_eval)
        eval_cameras = _load_cameras(self.datamanager.eval_dataset.cameras)

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=None,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=None,
            train_images=train_images,
            train_cameras=train_cameras,
            eval_images=eval_images,
            eval_cameras=eval_cameras,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        images_dict.pop("image_idx")  # wandb treat all items in image dict as image tensors
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_image_metrics(
        self,
        data_loader,
        image_prefix: str,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the dataset and get the average.

        Args:
            data_loader: the data loader to iterate over
            image_prefix: prefix to use for the saved image filenames
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            idx = 0
            for camera, batch in data_loader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )

        self.train()
        return metrics_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """

        self.eval()
        metrics_dict_list = []
        # assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullSequenceDatamanager))
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        start_time = None
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            # disable=False,
            disable=True,
            transient=True,
        ) as progress:
            if start_time is None:
                start_time = time()

            # self.model.register_eval_cameras()
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images, visible=True)
            iter_index = 0  # the iteration order of dataloader may be wrong
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                batch["camera_idx"] = batch["image_idx"]
                if camera.metadata is None:
                    camera.metadata = {"eval_idx": batch["image_idx"]}
                else:
                    camera.metadata["eval_idx"] = batch["image_idx"]
                iter_index += 1
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

                if output_path is not None:
                    idx = images_dict.get("image_idx", iter_index)  # use image index in batch item
                    for key, val in images_dict.items():
                        if torch.is_tensor(val):
                            if val.dtype == torch.uint8:
                                val = val.cpu().numpy()
                            elif val.dtype == torch.int32:
                                val = torch.clip(val, 0, 65535).cpu().numpy().astype(np.uint16)
                            else:
                                val = (val * 255).clip(0, 255).byte().cpu().numpy()
                            if len(val.shape) == 3:
                                if val.shape[-1] == 1:
                                    val = val[:, :, 0]
                                elif val.shape[-1] == 3:
                                    pass
                                else:
                                    continue
                            Image.fromarray(val).save(output_path / "{0:06d}-{1}.png".format(idx, key))
                        elif isinstance(val, open3d.geometry.PointCloud):
                            open3d.io.write_point_cloud(str(output_path / "points.ply"), val)

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                end_time = time()
        fps = iter_index / (end_time - start_time)
        CONSOLE.print(f"fps: {fps}")
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )

        if output_path is not None:
            if hasattr(self.model, "plot_cameras"):
                self.model.plot_cameras(os.path.dirname(str(output_path)), "pose")

        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        return {**datamanager_params, **model_params}
