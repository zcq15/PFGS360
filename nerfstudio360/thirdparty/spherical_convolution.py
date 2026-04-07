import numpy as np
import torch
import torch.nn as nn
from kornia.core.check import KORNIA_CHECK_SHAPE
from spherical_distortion.functional import mapped_convolution
from spherical_distortion.layer_utils import InterpolationType


def compute_axis(azimuth: torch.Tensor, polar: torch.Tensor):
    axis_z = torch.stack(
        [
            torch.sin(polar) * torch.sin(azimuth),
            -torch.cos(polar),
            torch.sin(polar) * torch.cos(azimuth),
        ],
        dim=-1,
    )
    axis_x = torch.stack(
        [
            torch.cos(azimuth),
            torch.zeros_like(azimuth),
            -torch.sin(azimuth),
        ],
        dim=-1,
    )

    axis_y = torch.stack(
        [
            torch.cos(polar) * torch.sin(azimuth),
            torch.sin(polar),
            torch.cos(polar) * torch.cos(azimuth),
        ],
        dim=-1,
    )
    return axis_x, axis_y, axis_z


# use opengl coordinates, (x, y, z) = (right, down, forward)
@torch.no_grad()
def compute_maps(height, width, kernel_size=11, stride=1, dilation=1, device="cuda", dtype=torch.float32):
    assert height * 2 == width
    if isinstance(kernel_size, int):
        ky = kernel_size
        kx = kernel_size
    elif isinstance(kernel_size, (list, tuple)):
        ky, kx = kernel_size[0], kernel_size[1]
    else:
        raise NotImplementedError

    azimuth = torch.linspace(-1 + 1 / width, 1 - 1 / width, width, dtype=dtype, device=device) * torch.pi
    polar = torch.linspace(1 / (2 * height), 1 - 1 / (2 * height), height, dtype=dtype, device=device) * torch.pi
    azimuth, polar = torch.meshgrid(azimuth[::stride], polar[::stride], indexing="xy")
    h, w = azimuth.shape[:2]

    axis_x, axis_y, axis_z = compute_axis(azimuth, polar)

    xx = torch.arange(0, kx, dtype=dtype, device=device) - kx // 2
    yy = torch.arange(0, ky, dtype=dtype, device=device) - ky // 2
    xx, yy = torch.meshgrid(xx, yy, indexing="xy")  # [kx, ky, 3]

    axis_x = axis_x.view(h, w, 1, 1, 3)
    axis_y = axis_y.view(h, w, 1, 1, 3)
    axis_z = axis_z.view(h, w, 1, 1, 3)

    xx = xx.view(1, 1, ky, kx, 1) * dilation * torch.pi / height
    yy = yy.view(1, 1, ky, kx, 1) * dilation * torch.pi / height

    xyz = axis_z + axis_x * xx + axis_y * yy  # [h, w, ky, kx, 3]
    xyz = xyz.reshape(h, w, ky * kx, 3)

    polar = torch.atan2(torch.sqrt(xyz[..., 0].square() + xyz[..., 2].square()), -xyz[..., 1])
    azimuth = torch.atan2(xyz[..., 0], xyz[..., 2])

    x_coords = (azimuth + torch.pi) / (torch.pi * 2) * width - 0.5
    y_coords = polar / torch.pi * height - 0.5

    grids = torch.stack([x_coords, y_coords], dim=-1)  # [h, w, k*k, 2]

    return grids.contiguous()  # [hk, wk, 2]


class SphericalConv2D(nn.Module):
    """SphereConv2D
    Note that this layer only support 3x3 filter
    """

    def __init__(
        self,
        in_c,
        out_c,
        kernel_size=11,
        bias=True,
        stride=1,
        dilation=1,
    ):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        if isinstance(kernel_size, int):
            ky = kernel_size
            kx = kernel_size
        elif isinstance(kernel_size, (list, tuple)):
            ky, kx = kernel_size[0], kernel_size[1]
        else:
            raise NotImplementedError
        self.weight = nn.Parameter(torch.Tensor(out_c, in_c, ky, kx))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter("bias", None)
        self.grid_shape = None
        self.grid = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        KORNIA_CHECK_SHAPE(x, ["B", "C", "H", "W"])
        if not self.grid_shape == tuple([x.shape[-2] // self.stride, x.shape[-1] // self.stride]):
            self.grid_shape = tuple([x.shape[-2] // self.stride, x.shape[-1] // self.stride])
            self.grid = compute_maps(
                height=x.shape[-2],
                width=x.shape[-1],
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                device=x.device,
            )
        if isinstance(self.kernel_size, int):
            ky = self.kernel_size
            kx = self.kernel_size
        elif isinstance(self.kernel_size, (list, tuple)):
            ky, kx = self.kernel_size[0], self.kernel_size[1]
        else:
            raise NotImplementedError
        weight = self.weight.view(self.out_c, self.in_c, ky * kx)
        bias = torch.zeros([self.out_c]).to(weight.device) if self.bias is None else self.bias
        y = mapped_convolution(x.contiguous(), weight, bias, self.grid, ky * kx, InterpolationType.BILINEAR)
        return y


def spherical_conv2d(x, weight, bias=None, stride=1, dilation=1):
    b, c, h, w = x.shape
    out_c, in_c, ky, kx = weight.shape
    assert c == in_c
    grid = compute_maps(h, w, [ky, kx], stride, dilation, device=x.device)
    weight = weight.view(out_c, in_c, ky * kx).contiguous()
    bias = torch.zeros([out_c]).to(weight.device) if bias is None else bias
    y = mapped_convolution(x.contiguous(), weight, bias, grid, ky * kx, InterpolationType.BILINEAR)
    return y
