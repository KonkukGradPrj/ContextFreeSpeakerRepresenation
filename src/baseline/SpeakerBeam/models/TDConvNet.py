###########################################################
# 모델에 쓰이는 기본적인 time-domain convolution net을 구현한 모듈 #
###########################################################

from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import warnings

from .norm import GlobLN

class _Chop1d(nn.Module):
    """To ensure the output length is the same as the input."""

    def __init__(self, chop_size):
        super().__init__()
        self.chop_size = chop_size

    def forward(self, x):
        return x[..., : -self.chop_size].contiguous()
    
class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        skip_out_chan (int): Number of channels in the skip convolution.
            If 0 or None, `Conv1DBlock` won't have any skip connections.
            Corresponds to the the block in v1 or the paper. The `forward`
            return res instead of [res, skip] in this case.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        padding (int): Padding of the depth-wise convolution.
        dilation (int): Dilation of the depth-wise convolution.
        norm_type (str, optional): Type of normalization to use. To choose from

            -  ``'gLN'``: global Layernorm.
            -  ``'cLN'``: channelwise Layernorm.
            -  ``'cgLN'``: cumulative global Layernorm.
            -  Any norm supported by :func:`~.norms.get`
        causal (bool, optional) : Whether or not the convolutions are causal


    References
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        in_chan,
        hid_chan,
        skip_out_chan,
        kernel_size,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super(Conv1DBlock, self).__init__()
        self.skip_out_chan = skip_out_chan
        conv_norm = GlobLN              ########## 일단 GlobLN으로 고정 // speakerbeam implementation에서도 default 사용
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(
            hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan
        )
        if causal:
            depth_conv1d = nn.Sequential(depth_conv1d, _Chop1d(padding))
        self.shared_block = nn.Sequential(
            in_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
            depth_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        if skip_out_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        r"""Input shape $(batch, feats, seq)$."""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_out_chan:
            return res_out
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out

class TDConvNet(nn.Module):
    """Temporal Convolutional network used in ConvTasnet.

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        causal (bool, optional) : Whether or not the convolutions are causal.

    References
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="relu",
        causal=False,
    ):
        super(TDConvNet, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = causal

        layer_norm = GlobLN(in_chan) ########## 일단 GlobLN으로 고정 // speakerbeam implementation에서도 default 사용
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                if not causal:
                    padding = (conv_kernel_size - 1) * 2**x // 2
                else:
                    padding = (conv_kernel_size - 1) * 2**x
                self.TCN.append(
                    Conv1DBlock(
                        bn_chan,
                        hid_chan,
                        skip_chan,
                        conv_kernel_size,
                        padding=padding,
                        dilation=2**x,
                        norm_type=norm_type,
                        causal=causal,
                    )
                )
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        self.output_act = nn.ReLU() ### speakerbeam에서 사용한 ReLU로 통일

    def forward(self, mixture_w):
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.TCN:
            # Common to w. skip and w.o skip architectures
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "out_chan": self.out_chan,
            "bn_chan": self.bn_chan,
            "hid_chan": self.hid_chan,
            "skip_chan": self.skip_chan,
            "conv_kernel_size": self.conv_kernel_size,
            "n_blocks": self.n_blocks,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "mask_act": self.mask_act,
            "causal": self.causal,
        }
        return config