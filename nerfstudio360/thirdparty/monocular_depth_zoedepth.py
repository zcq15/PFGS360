import warnings
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")


class ZoeDepth(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
        # torch.hub.help("isl-org/ZoeDepth", "ZoeD_NK", force_reload=True)
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
        self.zoedepth = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(device)
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, rgb):
        raise NotImplementedError

    def infer_tensor(self, rgb):  # [b, 3, h, w], [0, 1] -> [b, 1, h, w]
        rgb_device = rgb.device
        assert torch.is_tensor(rgb) and len(rgb.shape) == 4 and rgb.shape[1] == 3
        return self.zoedepth.infer(rgb.to(self.device)).clip(min=1e-6).to(rgb_device)

    @torch.no_grad()
    def infer_pillow(self, rgb):  # [h, w, 3], [0, 255] with np.uint8 -> [h, w]
        assert isinstance(rgb, (np.ndarray, Image.Image))
        if isinstance(rgb, np.ndarray):
            assert rgb.dtype == np.uint8 and len(rgb.shape) == 3 and rgb.shape[-1] == 3
        if isinstance(rgb, Image.Image):
            assert rgb.mode == "RGB"
        rgb_tensor = transforms.ToTensor()(rgb).unsqueeze(0).to(self.device)
        return self.zoedepth.infer(rgb_tensor).clip(min=1e-6).squeeze().cpu().numpy()

    def infer_gsplat(self, rgb):  # [h, w, 3], [0, 1] -> [h, w, 1]
        rgb_device = rgb.device
        assert torch.is_tensor(rgb)
        assert rgb.dtype == torch.float32 and len(rgb.shape) == 3 and rgb.shape[-1] == 3
        rgb_tensor = rgb.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.zoedepth.infer(rgb_tensor).clip(min=1e-6).squeeze().unsqueeze(-1).to(rgb_device)
