from nerfstudio360.engines.nerfstudio360_trainer import TrainerConfig

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.pixel_samplers import PixelSamplerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.schedulers import MultiStepSchedulerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import LoggingConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from gsplat360.optimizers import SelectiveAdam

from typing import Type, Tuple
import torch
from dataclasses import dataclass
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.configs.base_config import LocalWriterConfig


@dataclass
class SGDOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.SGD
    momentum: float = 0.9
    """The momentum to use."""

    # TODO: somehow make this more generic. i dont like the idea of overriding the setup function
    # but also not sure how to go about passing things into predefined torch objects.
    def setup(self, params) -> torch.optim.Optimizer:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        kwargs.pop("max_norm")
        kwargs.pop("eps")
        return self._target(params, **kwargs)


@dataclass
class RAdamBetasOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.RAdam
    betas: Tuple[float, float] = (0.9, 0.999)
    """The betas to use."""
    weight_decay: float = 0
    """The weight decay to use."""


@dataclass
class SelectiveAdamOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = SelectiveAdam
    betas: Tuple[float, float] = (0.9, 0.999)
    """The betas to use."""
    weight_decay: float = 0
    """The weight decay to use."""
    force_enable: bool = True
