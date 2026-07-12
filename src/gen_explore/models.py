"""Manages the two FLUX pipelines this project needs -- Kontext for
identity-preserving generation and FLUX.1-dev+ControlNet-Union for restyling
-- as mutually exclusive, lazily (re)loaded modes rather than keeping both
resident. Each alone uses ~11-12GB VRAM quantized; loading both at once would
sit right at the edge of a 24GB budget with no headroom, so `ModelBundle`
swaps one out before loading the other. Loading a transformer takes
tens of seconds, so callers should batch all Kontext-needing work together
and all restyle-needing work together, rather than interleaving calls -- see
scripts/generate_character_actions.py for the batching pattern.
"""

import gc

import torch
from controlnet_aux import HEDdetector
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import FluxControlNetModel, FluxControlNetPipeline, FluxKontextPipeline, FluxTransformer2DModel
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import T5EncoderModel

from gen_explore import config


class ModelBundle:
    def __init__(self, device: str = config.DEVICE):
        self.device = device
        self.dtype = getattr(torch, config.DTYPE_NAME)
        self.hed_detector = HEDdetector.from_pretrained(config.HED_ANNOTATOR_WEIGHTS).to(device)
        self._kontext_pipe = None
        self._controlnet_pipe = None

    def _quant_config(self) -> DiffusersBitsAndBytesConfig:
        return DiffusersBitsAndBytesConfig(
            load_in_4bit=config.QUANTIZE_4BIT, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self.dtype
        )

    def _text_quant_config(self) -> TransformersBitsAndBytesConfig:
        return TransformersBitsAndBytesConfig(
            load_in_4bit=config.QUANTIZE_4BIT, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self.dtype
        )

    def _unload_kontext(self) -> None:
        if self._kontext_pipe is not None:
            del self._kontext_pipe
            self._kontext_pipe = None
            torch.cuda.empty_cache()

    def _unload_controlnet(self) -> None:
        if self._controlnet_pipe is not None:
            del self._controlnet_pipe
            self._controlnet_pipe = None
            torch.cuda.empty_cache()

    def unload_all(self) -> None:
        """Frees every model this bundle may hold (both pipelines, the HED
        detector) so nothing is resident on GPU/CPU once a top-level api.py
        call returns -- standalone API functions call this in a `finally`
        block, regardless of whether generation succeeded."""
        self._unload_kontext()
        self._unload_controlnet()
        if self.hed_detector is not None:
            del self.hed_detector
            self.hed_detector = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def kontext(self) -> FluxKontextPipeline:
        if self._kontext_pipe is None:
            self._unload_controlnet()
            transformer = FluxTransformer2DModel.from_pretrained(
                config.KONTEXT_REPO,
                subfolder="transformer",
                quantization_config=self._quant_config(),
                torch_dtype=self.dtype,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                config.KONTEXT_REPO,
                subfolder="text_encoder_2",
                quantization_config=self._text_quant_config(),
                torch_dtype=self.dtype,
            )
            self._kontext_pipe = FluxKontextPipeline.from_pretrained(
                config.KONTEXT_REPO, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=self.dtype
            )
            self._kontext_pipe.enable_model_cpu_offload()
        return self._kontext_pipe

    @property
    def controlnet_pipe(self) -> FluxControlNetPipeline:
        if self._controlnet_pipe is None:
            self._unload_kontext()
            transformer = FluxTransformer2DModel.from_pretrained(
                config.FLUX_BASE_REPO,
                subfolder="transformer",
                quantization_config=self._quant_config(),
                torch_dtype=self.dtype,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                config.FLUX_BASE_REPO,
                subfolder="text_encoder_2",
                quantization_config=self._text_quant_config(),
                torch_dtype=self.dtype,
            )
            controlnet = FluxControlNetModel.from_pretrained(config.CONTROLNET_UNION_REPO, torch_dtype=self.dtype)
            self._controlnet_pipe = FluxControlNetPipeline.from_pretrained(
                config.FLUX_BASE_REPO,
                controlnet=controlnet,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                torch_dtype=self.dtype,
            )
            self._controlnet_pipe.enable_model_cpu_offload()
        return self._controlnet_pipe
