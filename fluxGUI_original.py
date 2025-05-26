import os
import torch
import gradio as gr
from einops import rearrange, repeat
from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan
from scipy.io import wavfile
import glob
import random
import numpy as np
import re
import datetime # Added
from data_processor import process_zip_file # Added

# Import necessary functions and classes
from utils import load_t5, load_clap
# Ensure this points to the refactored training script if train.py is replaced by train_refactored.py
# For now, assuming train.py is the entry point for RF class, or it's defined elsewhere.
# If RF is in train_refactored.py, this would be: from train_refactored import RF
try:
    from train import RF # Try importing from original train.py
except ImportError:
    # This assumes train_refactored.py exists in the same directory
    # And that it contains the RF class definition.
    from train_refactored import RF 

from constants import build_model

# Disable flash attention if not available
# Check for PyTorch version that supports this attribute.
if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(False)


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

def prepare(t5, clip, img, prompt): # This is for the generation tab
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device) # Ensure img_ids is on the same device
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    
    txt = t5(prompt) # Generate text embeddings
    
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=img.device) # Ensure txt_ids is on the same device

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
    print("Model unloaded.")

def load_model_action(model_name): # Renamed to avoid conflict with Gradio component
    global global_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    unload_current_model()
    
    if not model_name or model_name == "None":
        msg = "No model selected or model is 'None'. Skipping model loading."
        print(msg)
        return msg

    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        msg = f"Error: Model file not found at {model_path}"
        print(msg)
        return msg

    # Determine model size from filename (example logic, adjust as needed)
    if 'musicflow_b' in model_name: model_size = "base"
    elif 'musicflow_g' in model_name: model_size = "giant"
    elif 'musicflow_l' in model_name: model_size = "large"
    elif 'musicflow_s' in model_name: model_size = "small"
    else: model_size = "base" 
    
    print(f"Loading {model_size} model: {model_name} from {model_path}")
    
    try:
        global_model = build_model(model_size).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True) # Use device for map_location
        
        # Adjust state_dict loading based on common patterns (ema, model, or direct)
        if 'ema' in state_dict:
            global_model.load_state_dict(state_dict['ema'])
        elif 'model' in state_dict:
            global_model.load_state_dict(state_dict['model'])
        else:
            global_model.load_state_dict(state_dict)

        global_model.eval()
        setattr(global_model, 'model_path', model_path) # Store for reference using setattr
        msg = f"Model {model_name} loaded successfully."
        print(msg)
        return msg
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        global_model = None # Ensure model is None if loading failed
        return f"Error loading model {model_name}: {e}"


def load_resources():
    global global_t5, global_clap, global_vae, global_vocoder, global_diffusion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if global_t5 is None:
        print("Loading T5 model...")
        global_t5 = load_t5(device, max_length=256)
    if global_clap is None:
        print("Loading CLAP model...")
        global_clap = load_clap(device, max_length=256)
    if global_vae is None:
        print("Loading VAE...")
        global_vae = AutoencoderKL.from_pretrained('cvssp/audioldm2', subfolder="vae").to(device)
    if global_vocoder is None:
        print("Loading vocoder...")
        global_vocoder = SpeechT5HifiGan.from_pretrained('cvssp/audioldm2', subfolder="vocoder").to(device)
    if global_diffusion is None:
        print("Initializing diffusion...")
        global_diffusion = RF()
    print("Base resources checked/loaded successfully!")

def generate_music(prompt, seed, cfg_scale, steps, duration, progress=gr.Progress()):
    global global_model, global_t5, global_clap, global_vae, global_vocoder, global_diffusion
    
    if global_model is None:
        return "Please select and load a model first.", None
    
    if seed == 0: seed = random.randint(1, 1000000)
    print(f"Using seed: {seed}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    segment_duration = 10 
    num_segments = int(np.ceil(duration / segment_duration))
    all_waveforms = []

    for i in range(num_segments):
        progress((i+1) / num_segments, desc=f"Generating segment {i+1}/{num_segments}")
        torch.manual_seed(seed + i) 
        latent_size = (256, 16)
        conds_txt = [prompt]
        unconds_txt = ["low quality, gentle"] # Example unconditional prompt
        L = len(conds_txt)
        init_noise = torch.randn(L, 8, latent_size[0], latent_size[1]).to(device)
        img, conds = prepare(global_t5, global_clap, init_noise, conds_txt)
        _, unconds = prepare(global_t5, global_clap, init_noise, unconds_txt)

        with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=torch.cuda.is_available()):
            images = global_diffusion.sample_with_xps(global_model, img, conds=conds, null_cond=unconds, sample_steps=steps, cfg=cfg_scale)
        
        images = rearrange(images[-1], "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=128, w=8, ph=2, pw=2)
        latents = 1 / global_vae.config.scaling_factor * images
        mel_spectrogram = global_vae.decode(latents).sample
        x_i = mel_spectrogram[0]
        if x_i.dim() == 4: x_i = x_i.squeeze(1)
        waveform = global_vocoder(x_i)[0].cpu().float().detach().numpy()
        all_waveforms.append(waveform)

    final_waveform = np.concatenate(all_waveforms)
    sample_rate = 16000
    final_waveform = final_waveform[:int(duration * sample_rate)]
    
    progress(0.95, desc="Saving audio file") 
    output_dir = os.path.join(current_dir, 'generations')
    os.makedirs(output_dir, exist_ok=True)
    prompt_part = re.sub(r'[^\w\s-]', '', prompt)[:20].strip().replace(' ', '_')
    model_name_part = os.path.splitext(os.path.basename(getattr(global_model, 'model_path', 'unknown_model')))[0]
    base_filename = f"{prompt_part}_{seed}_{model_name_part}"
    output_path = os.path.join(output_dir, f"{base_filename}.wav")
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{base_filename}_{counter}.wav")
        counter += 1
    wavfile.write(output_path, sample_rate, final_waveform)
    progress(1.0, desc="Audio generation complete")
    return f"Generated with seed: {seed}. Saved to: {os.path.basename(output_path)}", output_path

# Load base resources at startup
load_resources()

model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt")) + glob.glob(os.path.join(MODELS_DIR, "*.safetensors"))
model_choices = [os.path.basename(f) for f in model_files if os.path.isfile(f)]
if not model_choices: model_choices = ["None"] # Add "None" if no models found
default_model = 'musicflow_b.pt' if 'musicflow_b.pt' in model_choices else model_choices[0]

theme = gr.themes.Monochrome(primary_hue="gray", secondary_hue="gray", neutral_hue="gray", radius_size=gr.themes.sizes.radius_sm)

# Placeholder handler for training
def handle_start_training(zip_file, model_ver, epochs, batch_size, lr, ckpt_every, accum_iter, num_workers, suffix):
    print("--- Training Job Started ---")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    status_message = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training process initiated.\n"
    
    if zip_file is None:
        status_message += "Error: No ZIP file provided. Please upload your dataset.\n"
        print("Error: No ZIP file provided.")
        return status_message, "", "Training aborted: No data."

    # zip_file is a SpooledTemporaryFile object, .name gives its temporary path
    print(f"Uploaded ZIP File Path (temp): {zip_file.name}") 
    print(f"Selected Model Architecture: {model_ver}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}")
    print(f"Checkpoint Every (steps): {ckpt_every}, Accumulation Iterations: {accum_iter}, Num Workers: {num_workers}")
    print(f"Custom Suffix: {suffix if suffix else 'None'}")

    status_message += f"Uploaded ZIP: {os.path.basename(zip_file.name)}\n" # Show original filename for clarity
    status_message += f"Model: {model_ver}, Epochs: {epochs}, Batch: {batch_size}, LR: {lr}\n"

    base_extraction_dir = os.path.join(current_dir, "training_data_uploads")
    os.makedirs(base_extraction_dir, exist_ok=True)
    
    # Create a more unique output JSON name using the suffix if provided
    json_file_prefix = f"manifest_{model_ver}"
    if suffix:
        json_file_prefix += f"_{suffix}"
    output_json_name = f"{json_file_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    status_message += f"Processing ZIP file...\n"
    print(f"Calling process_zip_file with: {zip_file.name}, {base_extraction_dir}, {output_json_name}")
    
    # process_zip_file expects a file path string, not a file object directly for the first argument.
    generated_json_path = process_zip_file(zip_file.name, base_extraction_dir, output_json_name=output_json_name)
    
    json_path_output_value = ""
    trained_model_output_value = "Training logic not fully implemented here yet."

    if generated_json_path:
        status_message += f"ZIP processed. Training JSON manifest created at: {generated_json_path}\n"
        json_path_output_value = generated_json_path
        # Placeholder: Here you would typically call the actual training function from train_refactored.py
        # e.g., from train_refactored import run_training_session
        # final_model_path = run_training_session(data_path=generated_json_path, ...)
        status_message += "Placeholder: Would now call run_training_session with the generated JSON.\n"
        status_message += f"To train manually, use: python train_refactored.py --data-path \"{generated_json_path}\" ... (other args)\n"
    else:
        status_message += "Error: Failed to process ZIP file or create training JSON. Check console for details.\n"
        json_path_output_value = "Error during ZIP processing. See console."

    print("--- Training Job Handler Finished ---")
    return status_message, json_path_output_value, trained_model_output_value


# Gradio Interface
with gr.Blocks(theme=theme) as iface:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>FluxMusic Interface</h1>
            <p>Generate music or train new models.</p>
        </div>
        """)
    
    with gr.Tabs():
        with gr.TabItem("Generation"):
            gr.Markdown("## Music Generation")
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=model_choices, label="Select Model", value=default_model)
                load_model_button = gr.Button("Load Selected Model")
            load_status_output = gr.Textbox(label="Model Load Status", interactive=False)
            
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your music prompt here...")
                seed = gr.Number(label="Seed (0 for random)", value=0, precision=0)
            
            with gr.Row():
                cfg_scale = gr.Slider(minimum=1.0, maximum=40.0, step=0.1, label="CFG Scale", value=7.5) # Common default
                steps = gr.Slider(minimum=10, maximum=200, step=1, label="Steps", value=50) # Adjusted default
                duration = gr.Number(label="Duration (seconds)", value=10, minimum=5, maximum=120, step=1) # Adjusted range
            
            generate_button = gr.Button("Generate Music")
            generation_status_output = gr.Textbox(label="Generation Status", interactive=False)
            output_audio = gr.Audio(type="filepath", label="Generated Audio")

            load_model_button.click(load_model_action, inputs=[model_dropdown], outputs=[load_status_output])
            generate_button.click(generate_music, inputs=[prompt, seed, cfg_scale, steps, duration], outputs=[generation_status_output, output_audio])

        with gr.TabItem("Training"):
            gr.Markdown("## FluxMusic Model Training")
            gr.Markdown("Upload a ZIP file containing your audio files (e.g., .wav, .mp3) and a `metadata.jsonl` file at its root. Each line in `metadata.jsonl` should be a JSON object like: `{\"audio_filename\": \"relative/path/to/audio.wav\", \"prompt\": \"your text prompt\"}`. Audio paths in the manifest should be relative to the ZIP root.")
            
            training_zip_upload = gr.File(label="Upload Audio ZIP (.zip)", file_types=['.zip'])
            
            model_versions = ["small", "base", "large", "giant"] 
            training_model_version = gr.Dropdown(choices=model_versions, label="Select Model Architecture", value="small")
            
            with gr.Row():
                training_epochs = gr.Number(label="Epochs", value=100, precision=0)
                training_batch_size = gr.Number(label="Batch Size", value=4, precision=0) # Kept small as per original
                training_learning_rate = gr.Number(label="Learning Rate", value=3e-5, format="%.0e") # Corrected format
            
            with gr.Row():
                training_ckpt_every = gr.Number(label="Save Checkpoint Every (steps)", value=10000, precision=0)
                training_accum_iter = gr.Number(label="Gradient Accumulation Steps", value=16, precision=0) # Matched original default
                training_num_workers = gr.Number(label="Dataloader Num Workers", value=2, precision=0)

            trained_model_name_suffix = gr.Textbox(label="Custom Suffix for Trained Model Name", placeholder="e.g., my_finetune (optional)")
            
            start_training_button = gr.Button("Start Training")
            
            training_status_output = gr.Textbox(label="Training Status", lines=10, interactive=False, placeholder="Training logs and status will appear here...")
            generated_json_path_output = gr.Textbox(label="Path to Generated Training JSON", interactive=False)
            final_trained_model_path_output = gr.Textbox(label="Path to Final Trained Model", interactive=False)

            start_training_button.click(
                fn=handle_start_training, 
                inputs=[
                    training_zip_upload, 
                    training_model_version, 
                    training_epochs, 
                    training_batch_size, 
                    training_learning_rate, 
                    training_ckpt_every, 
                    training_accum_iter, 
                    training_num_workers,
                    trained_model_name_suffix
                ], 
                outputs=[
                    training_status_output, 
                    generated_json_path_output, 
                    final_trained_model_path_output
                ]
            )

    # Load default model on startup if it exists (for generation tab)
    if default_model != "None" and os.path.exists(os.path.join(MODELS_DIR, default_model)):
        # This lambda will be called when the interface loads.
        # It should return the initial status message for the load_status_output.
        iface.load(lambda model_name_on_load: load_model_action(model_name_on_load), inputs=[gr.State(default_model)], outputs=[load_status_output])


if __name__ == "__main__":
    iface.launch()