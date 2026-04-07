from unik3d.models import UniK3D as UniK3DModel
from unik3d.utils.camera import Spherical
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class UniK3D(torch.nn.Module):
    def __init__(self, model="unik3d-vitl", device="cuda"):
        super().__init__()
        self.device = device
        self.unik3d_model = UniK3DModel.from_pretrained(f"lpiccinelli/{model}").to(device)
        self.register_buffer("mean", torch.Tensor(MEAN).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.Tensor(STD).view(1, 3, 1, 1).to(device))
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def infer_tensor(self, rgb):  # [b, 3, h, w], [0, 1] -> [b, 1, h, w]
        rgb_device = rgb.device
        assert torch.is_tensor(rgb) and len(rgb.shape) == 4 and rgb.shape[1] == 3
        (rgb.to(self.device) - self.mean) / self.std
        params = torch.tensor([1.0, 1.0, 1.0, 1.0, float(rgb.shape[2]), float(rgb.shape[3]), torch.pi, torch.pi / 2])
        camera = Spherical(params=params)
        out = self.unik3d_model.infer(rgb=rgb, camera=camera, normalize=False)
        # depth = out["points"].norm(dim=1, keepdim=True).clip(min=1e-3)
        # Unik3D has calculated the distance map as the norms of points
        depth = out["distance"].clip(min=1e-3)
        return depth.to(rgb_device)

    @torch.no_grad()
    def infer_pillow(self, rgb):  # [h, w, 3], [0, 255] with np.uint8 -> [h, w]
        assert isinstance(rgb, (np.ndarray, Image.Image))
        if isinstance(rgb, np.ndarray):
            assert rgb.dtype == np.uint8 and len(rgb.shape) == 3 and rgb.shape[-1] == 3
        if isinstance(rgb, Image.Image):
            assert rgb.mode == "RGB"
        rgb_tensor = transforms.ToTensor()(rgb).unsqueeze(0).to(self.device)
        return self.infer_tensor(rgb_tensor).clip(min=1e-3).squeeze().cpu().numpy()

    @torch.no_grad()
    def infer_gsplat(self, rgb):  # [h, w, 3], [0, 1] -> [h, w, 1]
        rgb_device = rgb.device
        assert torch.is_tensor(rgb)
        assert rgb.dtype == torch.float32 and len(rgb.shape) == 3 and rgb.shape[-1] == 3
        rgb_tensor = rgb.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.infer_tensor(rgb_tensor).clip(min=1e-3).squeeze().unsqueeze(-1).to(rgb_device)
