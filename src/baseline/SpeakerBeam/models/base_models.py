########################################################################################
# Speakerbeam base의 base모듈 MaskNet을 담당                                              #
# TimeDomainSpeakerBeam <- BaseEncoderMaskerDecoderInformed <- BaseEncoderMaskerDecoder#
########################################################################################

import torch
import torch.nn as nn
import warnings
from typing import Optional
from utils import pad_x_to_y, jitable_shape

def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x
    
def _shape_reconstructed(reconstructed, size):
    """Reshape `reconstructed` to have same size as `size`

    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform

    Returns:
        torch.Tensor: Reshaped waveform

    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed
    
class BaseModel(nn.Module):
    """Base class for serializable models.

    Defines saving/loading procedures, and separation interface to `separate`.
    Need to overwrite the `forward` and `get_model_args` methods.

    Models inheriting from `BaseModel` can be used by :mod:`asteroid.separate`
    and by the `asteroid-infer` CLI. For models whose `forward` doesn't go from
    waveform to waveform tensors, overwrite `forward_wav` to return
    waveform tensors.

    Args:
        sample_rate (float): Operating sample rate of the model.
        in_channels: Number of input channels in the signal.
            If None, no checks will be performed.
    """

    def __init__(self, sample_rate: float, in_channels: Optional[int] = 1):
        super().__init__()
        self.__sample_rate = sample_rate
        self.in_channels = in_channels

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def sample_rate(self):
        """Operating sample rate of the model (float)."""
        return self.__sample_rate

    @sample_rate.setter
    def sample_rate(self, new_sample_rate: float):
        warnings.warn(
            "Other sub-components of the model might have a `sample_rate` "
            "attribute, be sure to modify them for consistency.",
            UserWarning,
        )
        self.__sample_rate = new_sample_rate

    def forward_wav(self, wav, *args, **kwargs):
        """Separation method for waveforms.

        In case the network's `forward` doesn't have waveforms as input/output,
        overwrite this method to separate from waveform to waveform.
        Should return a single torch.Tensor, the separated waveforms.

        Args:
            wav (torch.Tensor): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.
        """
        return self(wav, *args, **kwargs)

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError


class BaseEncoderMaskerDecoder(BaseModel):
    """Base class for encoder-masker-decoder separation models.

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masker network.
        decoder (Decoder): Decoder instance.
        encoder_activation (Optional[str], optional): Activation to apply after encoder.
            See ``asteroid.masknn.activations`` for valid values.
    """

    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__(sample_rate=getattr(encoder, "sample_rate", None))
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = nn.Identity()

    def forward(self, wav):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)

    def forward_encoder(self, wav: torch.Tensor) -> torch.Tensor:
        """Computes time-frequency representation of `wav`.

        Args:
            wav (torch.Tensor): waveform tensor in 3D shape, time last.

        Returns:
            torch.Tensor, of shape (batch, feat, seq).
        """
        tf_rep = self.encoder(wav)
        return self.enc_activation(tf_rep)

    def forward_masker(self, tf_rep: torch.Tensor) -> torch.Tensor:
        """Estimates masks from time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).

        Returns:
            torch.Tensor: Estimated masks
        """
        return self.masker(tf_rep)

    def apply_masks(self, tf_rep: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
        """Applies masks to time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq) shape.
            est_masks (torch.Tensor): Estimated masks.

        Returns:
            torch.Tensor: Masked time-frequency representations.
        """
        return est_masks * tf_rep.unsqueeze(1)

    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        """Reconstructs time-domain waveforms from masked representations.

        Args:
            masked_tf_rep (torch.Tensor): Masked time-frequency representation.

        Returns:
            torch.Tensor: Time-domain waveforms.
        """
        return self.decoder(masked_tf_rep)

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = self.encoder.filterbank.get_config()
        masknet_config = self.masker.get_config()
        # Assert both dict are disjoint
        if not all(k not in fb_config for k in masknet_config):
            raise AssertionError(
                "Filterbank and Mask network config share common keys. Merging them is not safe."
            )
        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **masknet_config,
            "encoder_activation": self.encoder_activation,
        }
        return model_args