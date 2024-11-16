# Setup AniimateDiff-Lightning video generation pipeline
# Note: This takes up ~4GB of GPU RAM
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
# from diffusers.utils import export_to_video

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
# import moviepy.editor
import torch


def get_video_gen_pipeline(device, step):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    base = "emilianJR/epiCRealism"

    # Load Motion Low-Rank Adaptations (LoRAs), which determines the type of motion in the videos
    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))

    # Setup pipeline with base model and animateDiff-lightning motion adaptor
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)

    # Set motion type
    pipe.load_lora_weights(
        "guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out"
    )

    # Setup scheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

    return pipe
