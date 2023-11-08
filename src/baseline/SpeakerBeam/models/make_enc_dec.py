########################################################
# env_dec모듈을 활용하여 쉽게 enc,dec를 make할 수 있게 하는 모듈 #
########################################################

from .enc_dec import Filterbank, Encoder, Decoder
import torch
import torch.nn as nn

def make_enc_dec(
    fb_name,
    n_filters,
    kernel_size,
    stride=None,
    sample_rate=8000.0,
    who_is_pinv=None,
    padding=0,
    output_padding=0,
    **kwargs,
):
    """Creates congruent encoder and decoder from the same filterbank family.

    Args:
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``]. Can also be a class defined in a
            submodule in this subpackade (e.g. :class:`~.FreeFB`).
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.0.
        who_is_pinv (str, optional): If `None`, no pseudo-inverse filters will
            be used. If string (among [``'encoder'``, ``'decoder'``]), decides
            which of ``Encoder`` or ``Decoder`` will be the pseudo inverse of
            the other one.
        padding (int): Zero-padding added to both sides of the input.
            Passed to Encoder and Decoder.
        output_padding (int): Additional size added to one side of the output shape.
            Passed to Decoder.
        **kwargs: Arguments which will be passed to the filterbank class
            additionally to the usual `n_filters`, `kernel_size` and `stride`.
            Depends on the filterbank family.
    Returns:
        :class:`.Encoder`, :class:`.Decoder`
    """
    fb_class = FreeFB

    if who_is_pinv in ["dec", "decoder"]:
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        enc = Encoder(fb, padding=padding)
        # Decoder filterbank is pseudo inverse of encoder filterbank.
        dec = Decoder.pinv_of(fb)
    elif who_is_pinv in ["enc", "encoder"]:
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        dec = Decoder(fb, padding=padding, output_padding=output_padding)
        # Encoder filterbank is pseudo inverse of decoder filterbank.
        enc = Encoder.pinv_of(fb)
    else:
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        enc = Encoder(fb, padding=padding)
        # Filters between encoder and decoder should not be shared.
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        dec = Decoder(fb, padding=padding, output_padding=output_padding)
    return enc, dec


def register_filterbank(custom_fb):
    """Register a custom filterbank, gettable with `filterbanks.get`.

    Args:
        custom_fb: Custom filterbank to register.

    """
    if custom_fb.__name__ in globals().keys() or custom_fb.__name__.lower() in globals().keys():
        raise ValueError(f"Filterbank {custom_fb.__name__} already exists. Choose another name.")
    globals().update({custom_fb.__name__: custom_fb})

class FreeFB(Filterbank):
    """Free filterbank without any constraints. Equivalent to
    :class:`nn.Conv1d`.

    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.

    Attributes:
        n_feats_out (int): Number of output filters.

    References
        [1] : "Filterbank design for end-to-end speech separation". ICASSP 2020.
        Manuel Pariente, Samuele Cornell, Antoine Deleforge, Emmanuel Vincent.
    """

    def __init__(self, n_filters, kernel_size, stride=None, sample_rate=8000.0, **kwargs):
        super().__init__(n_filters, kernel_size, stride=stride, sample_rate=sample_rate)
        self._filters = nn.Parameter(torch.ones(n_filters, 1, kernel_size))
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    def filters(self):
        return self._filters