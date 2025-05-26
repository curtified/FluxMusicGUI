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

# Import necessary functions and classes from utils.py and constants.py
from utils import load_t5, load_clap 
from constants import build_model
# Import RF from train_refactored.py (or train.py if it's the final one)
try:
    from train_refactored import RF 
except ImportError:
    try:
        from train import RF # Fallback if train_refactored is not yet the final name
    except ImportError:
        print("WARNING: RF class could not be imported from train.py or train_refactored.py. Generation/Training might fail.")
        # Define a dummy RF class if not found, to allow UI to load
        class RF:
            def __init__(self, *args, **kwargs): pass
            def sample_with_xps(self, *args, **kwargs): return torch.randn(1,1,1) #dummy output


# Disable flash attention if not available
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

# --- Helper Functions for Generation (Copied from original fluxGUI.py) ---
def prepare(t5, clip, img, prompt):
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    
    txt = t5(prompt)
    
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=img.device)

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

def load_model_action(model_name): # Corrected name
    global global_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unload_current_model()
    if not model_name or model_name == "None":
        msg = "No model selected."
        print(msg)
        return msg
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        msg = f"Error: Model file not found: {model_path}"
        print(msg)
        return msg
    
    # Determine model size from filename (example logic)
    if 'musicflow_b' in model_name: model_size = "base"
    elif 'musicflow_g' in model_name: model_size = "giant"
    elif 'musicflow_l' in model_name: model_size = "large"
    elif 'musicflow_s' in model_name: model_size = "small"
    else: model_size = "base" 
    
    print(f"Loading {model_size} model: {model_name}")
    try:
        global_model = build_model(model_size).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        if 'ema' in state_dict: global_model.load_state_dict(state_dict['ema'])
        elif 'model' in state_dict: global_model.load_state_dict(state_dict['model'])
        else: global_model.load_state_dict(state_dict)
        global_model.eval()
        setattr(global_model, 'model_path', model_path) # Store model path
        msg = f"Model {model_name} loaded successfully."
        print(msg)
        return msg
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        global_model = None
        return f"Error loading model {model_name}: {e}"

def load_resources():
    global global_t5, global_clap, global_vae, global_vocoder, global_diffusion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if global_t5 is None: global_t5 = load_t5(device, max_length=256)
    if global_clap is None: global_clap = load_clap(device, max_length=256)
    if global_vae is None: global_vae = AutoencoderKL.from_pretrained('cvssp/audioldm2', subfolder="vae").to(device)
    if global_vocoder is None: global_vocoder = SpeechT5HifiGan.from_pretrained('cvssp/audioldm2', subfolder="vocoder").to(device)
    if global_diffusion is None: global_diffusion = RF()
    print("Base resources loaded.")

def generate_music(prompt, seed, cfg_scale, steps, duration, progress=gr.Progress()):
    global global_model, global_t5, global_clap, global_vae, global_vocoder, global_diffusion
    if global_model is None: return "Please select and load a model first.", None
    if seed == 0: seed = random.randint(1, 1000000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    segment_duration = 10 
    num_segments = int(np.ceil(duration / segment_duration))
    all_waveforms = []
    for i in range(num_segments):
        progress((i+1)/num_segments, desc=f"Generating segment {i+1}/{num_segments}")
        torch.manual_seed(seed + i) 
        latent_size = (256, 16); conds_txt = [prompt]; unconds_txt = ["low quality, gentle"]; L = len(conds_txt)
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
    final_waveform = np.concatenate(all_waveforms); sample_rate = 16000
    final_waveform = final_waveform[:int(duration * sample_rate)]
    output_dir = os.path.join(current_dir, 'generations'); os.makedirs(output_dir, exist_ok=True)
    prompt_part = re.sub(r'[^\w\s-]', '', prompt)[:20].strip().replace(' ', '_')
    model_name_part = os.path.splitext(os.path.basename(getattr(global_model, 'model_path', 'unknown')))[0]
    base_filename = f"{prompt_part}_{seed}_{model_name_part}"; output_path = os.path.join(output_dir, f"{base_filename}.wav")
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{base_filename}_{counter}.wav"); counter += 1
    wavfile.write(output_path, sample_rate, final_waveform)
    return f"Generated: {os.path.basename(output_path)} (Seed: {seed})", output_path

# --- Initial Calls ---
load_resources()
model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt")) + glob.glob(os.path.join(MODELS_DIR, "*.safetensors"))
model_choices = [os.path.basename(f) for f in model_files if os.path.isfile(f)]
if not model_choices: model_choices = ["None"]
default_model = 'musicflow_b.pt' if 'musicflow_b.pt' in model_choices else model_choices[0]
theme = gr.themes.Monochrome(primary_hue="gray",secondary_hue="gray",neutral_hue="gray",radius_size=gr.themes.sizes.radius_sm)

# --- Placeholder Training Handler ---
def handle_start_training(training_zip_upload, training_model_version, training_epochs, training_batch_size, training_learning_rate, training_ckpt_every, training_accum_iter, training_num_workers, trained_model_name_suffix):
    print("handle_start_training called") # Changed from (placeholder)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status_message = f"[{timestamp}] Training process initiated.\n"
    json_path_output = ""
    model_path_output = "Training logic not fully implemented here yet." # Default message

    if training_zip_upload is not None:
        status_message += f"Uploaded ZIP: {os.path.basename(training_zip_upload.name)}\n"
        status_message += f"Params: Model={training_model_version}, Epochs={training_epochs}, Batch={training_batch_size}, LR={training_learning_rate}, Suffix={trained_model_name_suffix}\n"
        status_message += f"Ckpt Every: {training_ckpt_every}, Accum Iter: {training_accum_iter}, Num Workers: {training_num_workers}\n"
        
        base_extraction_dir = os.path.join(current_dir, "training_data_uploads")
        os.makedirs(base_extraction_dir, exist_ok=True)
        
        json_file_prefix = f"manifest_{training_model_version}{'_' + trained_model_name_suffix if trained_model_name_suffix else ''}"
        output_json_name = f"{json_file_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        status_message += "Processing ZIP file...\n"
        generated_json_path = process_zip_file(training_zip_upload.name, base_extraction_dir, output_json_name=output_json_name)
        
        if generated_json_path:
            status_message += f"ZIP processed. Training JSON manifest created at: {generated_json_path}\n"
            json_path_output = generated_json_path
            # Placeholder for actual training call
            status_message += "Next step: Call run_training_session (not yet fully connected in UI).\n"
            print(f"Mock call would be: python train_refactored.py --data-path \"{generated_json_path}\" ... (plus other args from UI)")
        else:
            status_message += "Error processing ZIP file or manifest not found. Check console for details.\n"
            json_path_output = "Error during ZIP processing. See console."
    else:
        status_message += "Error: No ZIP file provided for training.\n"
            
    return status_message, json_path_output, model_path_output

# --- Gradio UI ---
with gr.Blocks(theme=theme) as iface:
    gr.Markdown("<div style='text-align: center;'><h1>FluxMusic Interface</h1><p>Generate music or train new models.</p></div>")
    
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
                cfg_scale = gr.Slider(minimum=1.0, maximum=40.0, step=0.1, label="CFG Scale", value=7.5)
                steps = gr.Slider(minimum=10, maximum=200, step=1, label="Steps", value=50)
                duration = gr.Number(label="Duration (seconds)", value=10, minimum=5, maximum=120, step=1)
            
            generate_button = gr.Button("Generate Music")
            generation_status_output = gr.Textbox(label="Generation Status", interactive=False)
            output_audio = gr.Audio(type="filepath", label="Generated Audio")

            load_model_button.click(load_model_action, inputs=[model_dropdown], outputs=[load_status_output])
            generate_button.click(generate_music, inputs=[prompt, seed, cfg_scale, steps, duration], outputs=[generation_status_output, output_audio])

        with gr.TabItem("Training"):
            gr.Markdown("## FluxMusic Model Training")
            gr.Markdown("Upload a ZIP file containing your audio files (e.g., .wav, .mp3) and a `metadata.jsonl` file at its root. Each line in `metadata.jsonl` should be a JSON object like: `{\"audio_filename\": \"relative/path/to/audio.wav\", \"prompt\": \"your text prompt\"}`. Audio paths in the manifest should be relative to the ZIP root.")
            
            with gr.Row():
                training_zip_upload = gr.File(label="Upload Audio ZIP (.zip)", file_types=['.zip'], scale=2)
                training_status_output = gr.Textbox(label="Training Status", lines=10, interactive=False, scale=3, placeholder="Status updates will appear here...") 

            with gr.Row():
                generated_json_path_output = gr.Textbox(label="Path to Generated Training JSON", interactive=False, scale=1)
                final_trained_model_path_output = gr.Textbox(label="Path to Final Trained Model", interactive=False, scale=1)

            gr.Markdown("### Training Configuration")
            with gr.Row():
                model_versions = ["small", "base", "large", "giant"] 
                training_model_version = gr.Dropdown(choices=model_versions, label="Select Model Architecture", value="small")
                trained_model_name_suffix = gr.Textbox(label="Custom Suffix for Output Dir", placeholder="e.g., my_finetune_run")

            with gr.Row():
                training_epochs = gr.Number(label="Epochs", value=100, precision=0)
                training_batch_size = gr.Number(label="Batch Size", value=4, precision=0)
                training_learning_rate = gr.Number(label="Learning Rate", value=3e-5, format="%.1e")
            
            with gr.Row():
                training_ckpt_every = gr.Number(label="Save Ckpt Every (steps)", value=10000, precision=0)
                training_accum_iter = gr.Number(label="Grad Accum Steps", value=4, precision=0) # Changed default from 16 to 4
                training_num_workers = gr.Number(label="Dataloader Num Workers", value=2, precision=0)
            
            start_training_button = gr.Button("Start Training Process (Generates JSON)", elem_id="start_training_button_id")
            
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

    # Load default model on startup for the Generation tab
    if default_model != "None" and os.path.exists(os.path.join(MODELS_DIR, default_model)):
        iface.load(lambda model_name_on_load: load_model_action(model_name_on_load), inputs=[gr.State(default_model)], outputs=[load_status_output])

if __name__ == "__main__":
    iface.launch()
