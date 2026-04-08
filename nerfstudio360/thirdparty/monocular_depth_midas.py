import warnings

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")


class MiDaS(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # from six.moves import urllib

        # proxy = urllib.request.ProxyHandler(
        #     {
        #         "http": "socks5://127.0.0.1:7890",
        #         "https": "socks5://127.0.0.1:7890",
        #         "all": "socks5://127.0.0.1:7890",
        #     }
        # )
        # opener = urllib.request.build_opener(proxy)
        # urllib.request.install_opener(opener)
        # torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", force_reload=True)
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(device)
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, rgb):
        raise NotImplementedError

    @torch.no_grad()
    def infer_tensor(self, rgb):  # [b, 3, h, w], [0, 1] -> [b, 1, h, w]
        rgb_device = rgb.device
        assert torch.is_tensor(rgb) and len(rgb.shape) == 4 and rgb.shape[1] == 3

        depth = []
        for b in range(rgb.shape[0]):
            # [h, w, 3], [0, 255], uint8
            rgb_numpy = (rgb[b] * 255.0).permute(1, 2, 0).clip(min=0, max=255).byte().cpu().numpy()
            depth.append(torch.from_numpy(self.infer_pillow(rgb_numpy)))
        depth = torch.stack(depth, dim=0).unsqueeze(dim=1).to(device=rgb_device)
        return depth

    @torch.no_grad()
    def infer_pillow(self, rgb):  # [h, w, 3], [0, 255] with np.uint8 -> [h, w]
        assert isinstance(rgb, (np.ndarray, Image.Image))
        if isinstance(rgb, np.ndarray):
            assert rgb.dtype == np.uint8 and len(rgb.shape) == 3 and rgb.shape[-1] == 3
        if isinstance(rgb, Image.Image):
            assert rgb.mode == "RGB"

        input_batch = self.midas_transforms(rgb).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        scale = 0.000305
        shift = 0.1378
        depth = scale * prediction + shift
        depth[depth < 1e-8] = 1e-8
        depth = 1.0 / depth
        return depth.clip(min=1e-6).cpu().numpy()  # [h, w]

    def infer_gsplat(self, rgb):  # [h, w, 3], [0, 1] -> [h, w, 1]
        rgb_device = rgb.device
        assert torch.is_tensor(rgb)
        assert rgb.dtype == torch.float32 and len(rgb.shape) == 3 and rgb.shape[-1] == 3
        rgb_numpy = (rgb * 255.0).clip(min=0, max=255).byte().cpu().numpy()
        depth_numpy = self.infer_pillow(rgb_numpy)
        return torch.from_numpy(depth_numpy).unsqueeze(dim=-1).to(device=rgb_device)
