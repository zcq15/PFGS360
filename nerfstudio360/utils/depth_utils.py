import torch
from da2 import DA2
from depth_anywhere.depth_anywhere import DepthAnywhere
from nerfstudio360.thirdparty.monocular_depth_midas import MiDaS
from nerfstudio360.thirdparty.monocular_depth_unik3d import UniK3D
from nerfstudio360.thirdparty.monocular_depth_zoedepth import ZoeDepth
from tqdm import tqdm


@torch.no_grad()
def generate_depth_sequence(image_sequence, model="midas"):  # [n, h, w, 3]
    assert (
        torch.is_tensor(image_sequence)
        and len(image_sequence.shape) == 4
        and image_sequence.shape[-1] == 3
        and image_sequence.dtype == torch.float32
    )
    models = {
        "midas": MiDaS,
        "zoedepth": ZoeDepth,
    }
    persp_depth_model = models[model](device="cuda" if torch.cuda.is_available() else "cpu")
    depth_sequence = []
    for idx in tqdm(range(len(image_sequence))):
        depth_sequence.append(persp_depth_model.infer_gsplat(image_sequence[idx]))
    depth_sequence = torch.stack(depth_sequence, dim=0)
    del persp_depth_model
    torch.cuda.empty_cache()
    return depth_sequence.clone().detach()  # [b, h, w, 1]


@torch.no_grad()
def generate_equir_depth_sequence(
    image_sequence,
    model="depth_anywhere",
):  # [n, h, w, 3]
    assert (
        torch.is_tensor(image_sequence)
        and len(image_sequence.shape) == 4
        and image_sequence.shape[-1] == 3
        and image_sequence.shape[1] * 2 == image_sequence.shape[2]
        and image_sequence.dtype == torch.float32
    )

    models = {
        "depth_anywhere": DepthAnywhere,
        "unik3d": UniK3D,
        "da2": DA2,
    }

    equir_depth_model = models[model](device="cuda" if torch.cuda.is_available() else "cpu")
    depth_sequence = []
    for idx in tqdm(range(len(image_sequence))):
        depth_sequence.append(equir_depth_model.infer_gsplat(image_sequence[idx]))
    depth_sequence = torch.stack(depth_sequence, dim=0)
    del equir_depth_model
    torch.cuda.empty_cache()
    return depth_sequence.clone().detach()  # [b, h, w, 1]


# copy from MiDaS and MonoSDF
@torch.no_grad()
def compute_scale_and_shift(prediction, target, mask=None):
    assert len(prediction.shape) == 3 and prediction.shape[-1] == 1
    assert len(target.shape) == 3 and target.shape[-1] == 1

    # only support shape [b, h, w]
    prediction = prediction.squeeze(-1).unsqueeze(0)
    target = target.squeeze(-1).unsqueeze(0)

    if mask is None:
        mask = torch.ones_like(prediction)
    else:
        mask = mask.type(prediction.dtype).squeeze(-1).unsqueeze(0)
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0.view(1), x_1.view(1)


@torch.no_grad()
def compute_aligned_depth(raw, reference, mask=None, inverse=False):

    selected = torch.logical_and(
        torch.logical_and(raw.squeeze(-1) > 0.1, raw.squeeze(-1) < 50.0),
        torch.logical_and(reference.squeeze(-1) > 0.1, reference.squeeze(-1) < 50.0),
    )
    if mask is not None:
        selected *= mask

    if inverse:
        raw_depth = 1.0 / raw.clip(min=1e-3, max=1e3)
        ref_depth = 1.0 / reference.clip(min=1e-3, max=1e3)
    else:
        raw_depth = raw.clip(min=1e-3, max=1e3)
        ref_depth = reference.clip(min=1e-3, max=1e3)

    scale, shift = compute_scale_and_shift(raw_depth[selected].view(-1, 1, 1), ref_depth[selected].view(-1, 1, 1))

    aligned_depth = raw_depth * scale.view(-1, 1, 1) + shift.view(-1, 1, 1)
    if inverse:
        aligned_depth = 1.0 / aligned_depth.clip(min=1e-3, max=1e3)
    else:
        aligned_depth = aligned_depth.clip(min=1e-3, max=1e3)

    return aligned_depth
