import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from nerfstudio.plugins.types import MethodSpecification

from nerfstudio360.dataparsers.openmvg_dataparser import OpenMVGDataParserConfig
from nerfstudio360.models.posefree_gaussian_splatting_360 import PoseFreeGSplat360ModelConfig
from nerfstudio360.pipelines.colmapfree_pipeline import ColmapFreePipelineConfig
from nerfstudio360.pipelines.fullimage_pipeline import FullImagePipelineConfig
from nerfstudio360.thirdparty.nerfstudio_component import *

pfgs360 = MethodSpecification(
    config=TrainerConfig(
        method_name="pfgs360",
        logging=LoggingConfig(
            steps_per_log=100,
            local_writer=LocalWriterConfig(enable=True, max_log_size=0),
        ),
        steps_per_eval_image=5000,
        steps_per_eval_batch=0,
        steps_per_save=5000,
        steps_per_eval_all_images=1000,
        max_num_iterations=1_000_000,
        mixed_precision=False,
        use_grad_scaler=False,
        pipeline=ColmapFreePipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=OpenMVGDataParserConfig(),
                cache_images="cpu",
            ),
            model=PoseFreeGSplat360ModelConfig(),
        ),
        optimizers={
            "means": {"optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15), "scheduler": None},
            "features_dc": {"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15), "scheduler": None},
            "features_rest": {"optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15), "scheduler": None},
            "opacities": {"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15), "scheduler": None},
            "scales": {"optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15), "scheduler": None},
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": SelectiveAdamOptimizerConfig(lr=1e-4, eps=1e-15, force_enable=True),
                "scheduler": None,
            },
            "eval_camera_opt": {
                "optimizer": SelectiveAdamOptimizerConfig(lr=1e-4, eps=1e-15, force_enable=True),
                "scheduler": None,
            },
            "calib": {"optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15), "scheduler": None},
            "placeholder": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15,
            quit_on_train_completion=True,
            image_format="png",
        ),
        vis="viewer",
    ),
    description="Implementation of Spherical Gaussian Splatting.",
)
