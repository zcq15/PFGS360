from typing import List, Optional, Tuple, Union

import torch
from kornia.core.check import KORNIA_CHECK_SHAPE
from nerfstudio360.thirdparty.spherical_convolution import compute_maps
from spherical_distortion.functional import mapped_convolution
from spherical_distortion.layer_utils import InterpolationType
from torch import Tensor


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.view([1, 1, 1, size])


def gaussian_filter(input: Tensor, win: Tensor, spherical: bool, maps: Tensor) -> Tensor:
    r"""Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert tuple(win.shape[:-1]) == tuple([1, 1, 1]) and len(win.shape) == 4
    B, C, H, W = input.shape
    output = input.view(B * C, 1, H, W)
    bias = torch.tensor([0], dtype=torch.float32, device=input.device)
    if spherical:
        output = mapped_convolution(
            output.contiguous(),
            win.view(1, 1, -1).contiguous(),
            bias,
            maps["X"].contiguous(),
            win.shape[-1],
            InterpolationType.BILINEAR,
        )
        output = mapped_convolution(
            output.contiguous(),
            win.view(1, 1, -1).contiguous(),
            bias,
            maps["Y"].contiguous(),
            win.shape[-1],
            InterpolationType.BILINEAR,
        )
    else:
        output = torch.nn.functional.conv2d(output, win, padding=[0, win.shape[-1] // 2])
        output = torch.nn.functional.conv2d(output, win.permute(0, 1, 3, 2), padding=[win.shape[-1] // 2, 0])
    return output.view([B, C, H, W])


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    spherical=False,
    maps=None,
) -> Tuple[Tensor, Tensor]:
    r"""Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win, spherical=spherical, maps=maps)
    mu2 = gaussian_filter(Y, win, spherical=spherical, maps=maps)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win, spherical=spherical, maps=maps) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win, spherical=spherical, maps=maps) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win, spherical=spherical, maps=maps) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    return ssim_map, cs_map


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
    spherical=False,
    maps=None,
) -> Tensor:
    r"""interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    if not len(X.shape) == 4:
        raise ValueError(f"Input images should be 4-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([1] + [1] * (len(X.shape) - 1))

    ssim_map, cs_map = _ssim(X, Y, data_range=data_range, win=win, K=K, spherical=spherical, maps=maps)

    if nonnegative_ssim:
        ssim_map = torch.relu(ssim_map)

    return ssim_map


class SphericalBlur(torch.nn.Module):
    def __init__(
        self,
        win_size: int = 11,
        win_sigma: float = 2.0,
        spherical=False,
    ) -> None:
        r"""class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SphericalBlur, self).__init__()
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.win = _fspecial_gauss_1d(win_size, win_sigma)
        self.spherical = spherical
        self.maps_shape = None
        self.maps = {"X": None, "Y": None}
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.eval()

    def update_maps(self, height, width, device):
        assert height * 2 == width
        if not tuple([height, width]) == self.maps_shape:
            self.maps_shape = tuple([height, width])
            self.maps["X"] = compute_maps(height, width, [1, self.win_size], device=device).contiguous()
            self.maps["Y"] = compute_maps(height, width, [self.win_size, 1], device=device).contiguous()

    def forward_tensor(self, X: Tensor) -> Tensor:
        if self.spherical:
            self.update_maps(X.shape[-2], X.shape[-1], X.device)

        self.win = self.win.to(X.device, dtype=X.dtype)

        X = gaussian_filter(X, self.win, spherical=self.spherical, maps=self.maps)

        return X

    def forward_gsplat(self, X: Tensor) -> Tensor:
        X = X.permute(2, 0, 1).unsqueeze(0)
        ans = self.forward_tensor(X)
        return ans.squeeze(0).permute(1, 2, 0)

    def forward(self, X: Tensor) -> Tensor:
        if len(X.shape) == 4:
            KORNIA_CHECK_SHAPE(X, ["B", "C", "H", "W"])
            return self.forward_tensor(X)
        elif len(X.shape) == 3:
            KORNIA_CHECK_SHAPE(X, ["H", "W", "C"])
            return self.forward_gsplat(X)
        else:
            print("please call forward_gsplat for HWC or forward_tensor for BCHW")
            raise NotImplementedError
