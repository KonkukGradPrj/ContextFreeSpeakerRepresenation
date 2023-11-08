import torch
import torch.nn as nn


def jitable_shape(tensor):
    """Gets shape of ``tensor`` as ``torch.Tensor`` type for jit compiler

    .. note::
        Returning ``tensor.shape`` of ``tensor.size()`` directly is not torchscript
        compatible as return type would not be supported.

    Args:
        tensor (torch.Tensor): Tensor

    Returns:
        torch.Tensor: Shape of ``tensor``
    """
    return torch.tensor(tensor.shape)

def pad_x_to_y(x: torch.Tensor, y: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Right-pad or right-trim first argument to have same size as second argument

    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad `x` to.
        axis (int): Axis to pad on.

    Returns:
        torch.Tensor, `x` padded to match `y`'s shape.
    """
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return nn.functional.pad(x, [0, inp_len - output_len])

