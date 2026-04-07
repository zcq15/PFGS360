from nerfstudio.engine.trainer import Trainer
from typing import DefaultDict, Dict, List, Literal, Optional, Tuple, Type, cast
from nerfstudio.utils import profiler
import torch
import functools

from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.engine.trainer import TrainerConfig
from pathlib import Path

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
TORCH_DEVICE = str


def set_timestamp(self) -> None:
    """Dynamically set the experiment timestamp"""
    self.timestamp = None


def get_base_dir(self) -> Path:
    """Retrieve the base directory to set relative paths"""
    # check the experiment and method names
    assert self.method_name is not None, "Please set method name in config or via the cli"
    self.set_experiment_name()
    if self.pipeline.datamanager.data is not None:
        datapath = Path(self.pipeline.datamanager.data)
    else:
        datapath = Path(self.pipeline.datamanager.dataparser.data)
    datapath = datapath.parent if datapath.is_file() else datapath
    if getattr(self.pipeline, "suffix", None) is None:
        return Path(f"{self.output_dir}/{self.experiment_name}/{str(datapath.stem)}/{self.method_name}")
    else:
        return Path(
            f"{self.output_dir}/{self.experiment_name}/{str(datapath.stem)}/{self.method_name}-{self.pipeline.suffix}"
        )


ExperimentConfig.set_timestamp = set_timestamp
ExperimentConfig.get_base_dir = get_base_dir
TrainerConfig.set_timestamp = set_timestamp
TrainerConfig.get_base_dir = get_base_dir


@profiler.time_function
def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
    """Run one iteration with a batch of inputs. Returns dictionary of model losses.

    Args:
        step: Current training step.
    """

    needs_zero = [
        group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
    ]
    self.optimizers.zero_grad_some(needs_zero)
    cpu_or_cuda_str: str = self.device.split(":")[0]
    cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

    with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
        _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
        loss = functools.reduce(torch.add, loss_dict.values())
    self.grad_scaler.scale(loss).backward()  # type: ignore
    needs_step = [
        group
        for group in self.optimizers.parameters.keys()
        if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
    ]
    for name, param in self.pipeline.model.named_parameters():
        if param.requires_grad and getattr(param, "grad", None) is not None:
            param.grad.nan_to_num_(0, 0, 0)
            if "pose_adjustment" in name:
                torch.nn.utils.clip_grad_value_(param, clip_value=1e-2)
    self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

    if self.config.log_gradients:
        total_grad = 0
        for tag, value in self.pipeline.model.named_parameters():
            assert tag != "Total"
            if value.grad is not None:
                grad = value.grad.norm()
                metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                total_grad += grad

        metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

    scale = self.grad_scaler.get_scale()
    self.grad_scaler.update()
    # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
    if scale <= self.grad_scaler.get_scale():
        self.optimizers.scheduler_step_all(step)

    # Merging loss and metrics dict into a single output.
    return loss, loss_dict, metrics_dict  # type: ignore


Trainer.train_iteration = train_iteration
