"""Model IDs and generation parameters for the candidate-line pipeline.

Migrated from SD1.5 to FLUX: SD1.5 produced duplicate subjects and malformed
anatomy that prompt engineering couldn't fix, and its IP-Adapter-based
character consistency broke specifically on poses meaningfully different from
the reference. FLUX Kontext (image-conditioned edit model, purpose-built for
character preservation) replaces the identity-anchor + IP-Adapter mechanism
entirely -- it handles both "generate fresh" (fed a blank white canvas, since
Kontext always expects an `image=`) and "generate a new pose for this
character" (fed the real identity anchor) through the same call shape, so
there's no more "load IP-Adapter only after the reference-free generation"
ordering constraint.

Restyling into construction-sketch linework now goes through FLUX.1-dev +
`Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`, using soft-edge (HED)
conditioning -- there is no genuinely diffusers-native scribble ControlNet for
FLUX.1-dev (checked five community options; four were ComfyUI-only or targeted
a different base model). Soft-edge is the closest real substitute, not a
perfect match for "scribble" -- flagged as a design compromise.
"""

KONTEXT_REPO = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX_BASE_REPO = "black-forest-labs/FLUX.1-dev"
CONTROLNET_UNION_REPO = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
HED_ANNOTATOR_WEIGHTS = "lllyasviel/Annotators"

DEVICE = "cuda"
DTYPE_NAME = "bfloat16"

# NF4 quantization (bitsandbytes) is required for both the Kontext and
# FLUX.1-dev transformers to fit alongside their T5-XXL text encoders in 24GB
# VRAM -- unquantized Kontext alone has been reported to need ~31.5GB.
QUANTIZE_4BIT = True

SKETCH_STYLE_SUFFIX = "as a simple, clean pencil sketch on a plain white background, no color, no shading"

# Characters were coming out either as heavily cross-hatched/shaded renderings
# (a single filled silhouette once vectorized -- no separate lines to choose
# between) or, when pushed toward "clean outline/line art", as tightly
# finished illustrations (too resolved, no room left for the human to clean
# up). The actual target is the classic Disney "ball-and-cylinder" animator's
# construction sketch: loose, scribbled, overlapping circles/loops indicating
# the volume of each body part -- rough and searching, not finished. The
# background's style is deliberately left untouched (it's already working).
CHARACTER_STYLE_SUFFIX = (
    "redrawn as a rough, loose construction sketch with scribbled gestural lines, "
    "animator's gesture drawing, unfinished, black and white, on a plain white background, no shading"
)

ROLE_STYLE_SUFFIX = {
    "background": SKETCH_STYLE_SUFFIX,
    # Locks in a recognizable subject/pose before the construction-style
    # restyling pass below -- deliberately the same neutral style as
    # "background" rather than "character", so the strong construction-sketch
    # language doesn't fight the subject on the very first pass.
    "character_subject": SKETCH_STYLE_SUFFIX,
    "character": CHARACTER_STYLE_SUFFIX,
    "fx": SKETCH_STYLE_SUFFIX,
}

# Kontext is guidance-distilled like FLUX.1-dev -- no meaningful negative_prompt,
# only guidance_scale. Used for both fresh generation (blank-canvas trick) and
# identity-conditioned pose generation.
KONTEXT_PARAMS = dict(
    guidance_scale=3.5,
    num_inference_steps=20,
)

# Restyle step: FLUX.1-dev + ControlNet-Union (soft-edge mode). The actual
# goal is multiple genuine outline OPTIONS per body part for cleanup
# selection, not a particular "scribbly" texture -- and a multi-seed sweep
# showed that chasing looser conditioning for scribbliness comes at the cost
# of consistency: below ~0.65 control_guidance_end the linework gets looser
# but the overall composition (desk shape, prop placement) can drift between
# candidates; above it, most seeds land on a consistent composition but some
# settle into a fully rendered/shaded look instead of staying line-based
# (unstable per-seed above ~0.75). 0.6/0.9 was the most reliably consistent
# combination found. Background keeps the original, tighter/tested values --
# it's already working, this change is deliberately scoped to characters only.
_BASELINE_RESTYLE_PARAMS = dict(
    controlnet_conditioning_scale=0.7,
    control_guidance_end=0.8,
    num_inference_steps=30,
    guidance_scale=3.5,
)
_CHARACTER_RESTYLE_PARAMS = dict(
    controlnet_conditioning_scale=0.6,
    control_guidance_end=0.9,
    num_inference_steps=30,
    guidance_scale=3.5,
)
ROLE_RESTYLE_PARAMS = {
    "background": _BASELINE_RESTYLE_PARAMS,
    "character": _CHARACTER_RESTYLE_PARAMS,
    "fx": _BASELINE_RESTYLE_PARAMS,
}

NUM_CANDIDATES = 3
CANDIDATE_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"]

# FLUX is trained/tuned for 1024x1024 (and multiples of 16) -- SD1.5's 768 was
# a different model's sweet spot, not a fixed requirement of this pipeline.
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
