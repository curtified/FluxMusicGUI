import os
import torch
import gradio as gr
from einops import rearrange, repeat
from diffusers import AutoencoderKL
from torch.cuda import is_available
from transformers import SpeechT5HifiGan
from scipy.io import wavfile
import glob
import random
import numpy as np
import re

# Import necessary functions and classes
from utils import load_t5, load_clap
from train import RF
from constants import build_model

# Disable flash attention if not available
# torch.backends.cuda.enable_flash_sdp(False)

# Global variables to store loaded models and resources
global_model = None
global_t5 = None
global_clap = None
global_vae = None
global_vocoder = None
global_diffusion = None

# Set the models directory relative to the script location
current_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(current_dir, "models")


def prepare(t5, clip, img, prompt):
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]

    # Generate text embeddings
    txt = t5(prompt)

    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return img, {
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "y": vec.to(img.device),
    }


def unload_current_model():
    global global_model
    if global_model is not None:
        del global_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        global_model = None


def load_model(model_name):
    global global_model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unload_current_model()

    # Determine model size from filename
    if "musicflow_b" in model_name:
        model_size = "base"
    elif "musicflow_g" in model_name:
        model_size = "giant"
    elif "musicflow_l" in model_name:
        model_size = "large"
    elif "musicflow_s" in model_name:
        model_size = "small"
    else:
        model_size = "base"  # Default to base if unrecognized

    print(f"Loading {model_size} model: {model_name}")

    model_path = os.path.join(MODELS_DIR, model_name)
    global_model = build_model(model_size).to(device)
    state_dict = torch.load(
        model_path, map_location=lambda storage, loc: storage, weights_only=True
    )
    global_model.load_state_dict(state_dict["ema"])
    global_model.eval()
    global_model.model_path = model_path


def load_resources():
    global global_t5, global_clap, global_vae, global_vocoder, global_diffusion

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading T5 and CLAP models...")
    global_t5 = load_t5(device, max_length=256)
    global_clap = load_clap(device, max_length=256)

    print("Loading VAE and vocoder...")
    global_vae = AutoencoderKL.from_pretrained("cvssp/audioldm2", subfolder="vae").to(
        device
    )
    global_vocoder = SpeechT5HifiGan.from_pretrained(
        "cvssp/audioldm2", subfolder="vocoder"
    ).to(device)

    print("Initializing diffusion...")
    global_diffusion = RF()

    print("Base resources loaded successfully!")


def generate_music(prompt, seed, cfg_scale, steps, duration, progress=gr.Progress()):
    global global_model, global_t5, global_clap, global_vae, global_vocoder, global_diffusion

    if global_model is None:
        return "Please select a model first.", None

    if seed == 0:
        seed = random.randint(1, 1000000)
    print(f"Using seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    # Calculate the number of segments needed for the desired duration
    segment_duration = 10  # Each segment is 10 seconds
    num_segments = int(np.ceil(duration / segment_duration))

    all_waveforms = []

    for i in range(num_segments):
        progress(i / num_segments, desc=f"Generating segment {i+1}/{num_segments}")

        # Use the same seed for all segments
        torch.manual_seed(
            seed + i
        )  # Add i to slightly vary each segment while maintaining consistency

        latent_size = (256, 16)
        conds_txt = [prompt]
        unconds_txt = ["low quality, gentle"]
        L = len(conds_txt)

        init_noise = torch.randn(L, 8, latent_size[0], latent_size[1]).to(device)

        img, conds = prepare(global_t5, global_clap, init_noise, conds_txt)
        _, unconds = prepare(global_t5, global_clap, init_noise, unconds_txt)

        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda"):
                images = global_diffusion.sample_with_xps(
                    global_model,
                    img,
                    conds=conds,
                    null_cond=unconds,
                    sample_steps=steps,
                    cfg=cfg_scale,
                )
        else:
            with torch.autocast(device_type="cpu"):
                images = global_diffusion.sample_with_xps(
                    global_model,
                    img,
                    conds=conds,
                    null_cond=unconds,
                    sample_steps=steps,
                    cfg=cfg_scale,
                )

        images = rearrange(
            images[-1],
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=128,
            w=8,
            ph=2,
            pw=2,
        )

        latents = 1 / global_vae.config.scaling_factor * images
        mel_spectrogram = global_vae.decode(latents).sample

        x_i = mel_spectrogram[0]
        if x_i.dim() == 4:
            x_i = x_i.squeeze(1)
        waveform = global_vocoder(x_i)
        waveform = waveform[0].cpu().float().detach().numpy()

        all_waveforms.append(waveform)

    # Concatenate all waveforms
    final_waveform = np.concatenate(all_waveforms)

    # Trim to exact duration
    sample_rate = 16000
    final_waveform = final_waveform[: int(duration * sample_rate)]

    progress(0.9, desc="Saving audio file")

    # Create 'generations' folder in the current directory
    output_dir = os.path.join(current_dir, "generations")
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    prompt_part = re.sub(r"[^\w\s-]", "", prompt)[:10].strip().replace(" ", "_")
    model_name = os.path.splitext(os.path.basename(global_model.model_path))[0]
    model_suffix = "_mf_b" if model_name == "musicflow_b" else f"_{model_name}"
    base_filename = f"{prompt_part}_{seed}{model_suffix}"
    output_path = os.path.join(output_dir, f"{base_filename}.wav")

    # Check if file exists and add numerical suffix if needed
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{base_filename}_{counter}.wav")
        counter += 1

    wavfile.write(output_path, sample_rate, final_waveform)

    progress(1.0, desc="Audio generation complete")
    return f"Generated with seed: {seed}", output_path


# Load base resources at startup
load_resources()

# Get list of .pt files in the models directory
model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
model_choices = [os.path.basename(f) for f in model_files]

# Ensure 'musicflow_b.pt' is the default choice if it exists
default_model = "musicflow_b.pt"
if default_model in model_choices:
    model_choices.remove(default_model)
    model_choices.insert(0, default_model)
    global_model = default_model


# Set up dark grey theme
theme = gr.themes.Monochrome(
    primary_hue="gray",
    secondary_hue="gray",
    neutral_hue="gray",
    radius_size=gr.themes.sizes.radius_sm,
)

# Gradio Interface
with gr.Blocks(theme=theme, analytics_enabled=False) as iface:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>FluxMusic Generator</h1>
            <p>Generate music based on text prompts using FluxMusic model.</p>
        </div>
        """
    )

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=model_choices,
            label="Select Model",
            value=default_model if default_model in model_choices else model_choices[0],
        )

    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
        seed = gr.Number(label="Seed", value=0)

    with gr.Row():
        cfg_scale = gr.Slider(
            minimum=1, maximum=40, step=0.1, label="CFG Scale", value=20
        )
        steps = gr.Slider(minimum=10, maximum=200, step=1, label="Steps", value=100)
        duration = gr.Number(
            label="Duration (seconds)", value=10, minimum=10, maximum=300, step=1
        )

    generate_button = gr.Button("Generate Music")
    output_status = gr.Textbox(label="Generation Status")
    output_audio = gr.Audio(type="filepath")

    def on_model_change(model_name):
        load_model(model_name)

    model_dropdown.change(on_model_change, inputs=[model_dropdown])
    generate_button.click(
        generate_music,
        inputs=[prompt, seed, cfg_scale, steps, duration],
        outputs=[output_status, output_audio],
    )

    # Load default model on startup
    default_model_path = os.path.join(MODELS_DIR, default_model)
    if os.path.exists(default_model_path):
        iface.load(lambda: load_model(default_model), inputs=None, outputs=None)

if __name__ == "__main__":
    iface.launch()
