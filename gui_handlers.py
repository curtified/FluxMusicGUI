# gui_handlers.py
# This file will contain the backend logic for the Gradio UI.

import os
import torch
import gradio as gr # For gr.Progress, though direct UI manipulation is avoided here
import einops
from einops import rearrange, repeat # Explicitly import for prepare
import numpy as np
from scipy.io import wavfile # For saving generated audio in generate_music
import re # For filename sanitization in generate_music
import uuid # For unique naming in handle_start_training
import datetime # For timestamp naming in handle_start_training
import shutil # For file operations if needed (e.g. data_processor)

# Imports from other project modules (assuming they are in PYTHONPATH)
# These will be used by functions moved into this file later.
from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan

# Attempt to import project-specific modules.
# If these fail, it indicates a path or file naming issue to be resolved.
try:
    from utils import load_t5, load_clap
except ImportError:
    print("Warning: Could not import 'load_t5', 'load_clap' from 'utils'. Ensure utils.py is accessible.")
    # Define placeholders if needed, or let subsequent errors occur if critical
    def load_t5(*args, **kwargs): raise NotImplementedError("load_t5 not imported")
    def load_clap(*args, **kwargs): raise NotImplementedError("load_clap not imported")

try:
    from constants import build_model
except ImportError:
    print("Warning: Could not import 'build_model' from 'constants'. Ensure constants.py is accessible.")
    def build_model(*args, **kwargs): raise NotImplementedError("build_model not imported")

try:
    from data_processor import process_zip_file
except ImportError:
    print("Warning: Could not import 'process_zip_file' from 'data_processor'. Ensure data_processor.py is accessible.")
    def process_zip_file(*args, **kwargs): raise NotImplementedError("process_zip_file not imported")

try:
    from train import RF, run_training_session # RF for diffusion, run_training_session for training
except ImportError:
    print("Warning: Could not import RF or run_training_session from 'train'. Ensure train.py is accessible.")
    class RF: 
        def __init__(self, *args, **kwargs): pass
        def sample_with_xps(self, *args, **kwargs): return torch.randn(1,1,1,1)
    def run_training_session(*args, **kwargs): 
        yield {'type':'error', 'message':'Training script (train.py or train_refactored.py) not found.'}


# Module-level constants (will be populated and used by moved functions)
current_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(current_dir, "models") # Assuming models are in 'models' subdirectory relative to this file

# Global model variables (will be populated by load_resources and load_model_action)
# These are kept in this handlers file as they represent the state managed by these handlers.
global_model = None
global_t5 = None
global_clap = None
global_vae = None
global_vocoder = None
global_diffusion = None # From original fluxGUI.py

print(f"gui_handlers.py initialized. current_dir: {current_dir}, MODELS_DIR: {MODELS_DIR}")


def prepare(t5, clip, img, prompt_in): 
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt_in, str):
        bs = len(prompt_in)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device) 
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt_in, str):
        prompt_in = [prompt_in]
    
    txt = t5(prompt_in) 
    
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=img.device) 

    vec = clip(prompt_in)
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
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
            torch.mps.empty_cache()
        global_model = None 
    print("Model unloaded and cache cleared if applicable.")

def load_model_action(model_name):
    global global_model, MODELS_DIR 
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"load_model_action using device: {device}")

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
        setattr(global_model, 'model_path', model_path) 
        msg = f"Model {model_name} loaded successfully on {device}."
        print(msg)
        return msg
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        global_model = None
        return f"Error loading model {model_name}: {e}"

def load_resources():
    global global_t5, global_clap, global_vae, global_vocoder, global_diffusion, current_dir

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available(): 
        device = "cuda"
    else:
        device = "cpu"
    print(f"load_resources using device: {device}")
    
    if global_t5 is None:
        print("Loading T5 model...")
        global_t5 = load_t5(device, max_length=256)
    if global_clap is None:
        print("Loading CLAP model...")
        global_clap = load_clap(device, max_length=256)
    
    if global_vae is None:
        vae_local_path = os.path.join(current_dir, "audioldm2", "vae")
        if os.path.isdir(vae_local_path):
            print(f"Loading VAE from local path: {vae_local_path}")
            global_vae = AutoencoderKL.from_pretrained(vae_local_path).to(device)
        else:
            print(f"Local VAE path {vae_local_path} not found. Loading from 'cvssp/audioldm2'.")
            global_vae = AutoencoderKL.from_pretrained('cvssp/audioldm2', subfolder="vae").to(device)

    if global_vocoder is None:
        vocoder_local_path = os.path.join(current_dir, "audioldm2", "vocoder")
        if os.path.isdir(vocoder_local_path):
            print(f"Loading Vocoder from local path: {vocoder_local_path}")
            global_vocoder = SpeechT5HifiGan.from_pretrained(vocoder_local_path).to(device)
        else:
            print(f"Local Vocoder path {vocoder_local_path} not found. Loading from 'cvssp/audioldm2'.")
            global_vocoder = SpeechT5HifiGan.from_pretrained('cvssp/audioldm2', subfolder="vocoder").to(device)

    if global_diffusion is None:
        print("Initializing diffusion (RF class)...")
        global_diffusion = RF() 
        
    print("Base resources checked/loaded successfully!")

def generate_music(prompt, seed, cfg_scale, steps, duration, progress=gr.Progress()):
    global global_model, global_t5, global_clap, global_vae, global_vocoder, global_diffusion, current_dir
    
    if global_model is None:
        return "Please select and load a model first.", None
    
    if seed == 0: 
        seed = random.randint(1, 1000000)
    print(f"Using seed: {seed}")
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"generate_music using device: {device}")

    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    segment_duration = 10 
    num_segments = int(np.ceil(duration / segment_duration))
    all_waveforms = []

    for i in range(num_segments):
        if progress is not None: 
            progress((i+1)/num_segments, desc=f"Generating segment {i+1}/{num_segments}")
        
        torch.manual_seed(seed + i) 
        latent_size = (256, 16)
        conds_txt = [prompt]
        unconds_txt = ["low quality, gentle"] 
        L = len(conds_txt)
        init_noise = torch.randn(L, 8, latent_size[0], latent_size[1]).to(device)
        
        img, conds = prepare(global_t5, global_clap, init_noise, conds_txt)
        _, unconds = prepare(global_t5, global_clap, init_noise, unconds_txt)

        autocast_device_type = 'cuda' if device == 'cuda' else 'cpu'
        with torch.autocast(device_type=autocast_device_type, enabled=(device == 'cuda')): 
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
    
    if progress is not None: progress(0.95, desc="Saving audio file") 
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
    if progress is not None: progress(1.0, desc="Audio generation complete")
    return f"Generated with seed: {seed}. Saved to: {os.path.basename(output_path)}", output_path

def handle_start_training(training_zip_upload, training_model_version, training_epochs, training_batch_size, training_learning_rate, training_ckpt_every, training_accum_iter, training_num_workers, trained_model_name_suffix):
    # Access module-level current_dir, or ensure it's passed/accessible
    # global current_dir # Not needed if current_dir is already module-level and not reassigned

    json_path_for_ui = ""
    model_path_for_ui = ""
    training_log = []

    def update_status(message, is_error=False):
        # Nonlocal training_log if this were nested deeper and training_log was from outer scope;
        # here it's in the same scope or can be passed if made a separate helper.
        prefix = "ERROR: " if is_error else "INFO: "
        log_entry = prefix + str(message) # Ensure message is string
        training_log.append(log_entry)
        # Keep only the last 20 lines for display to prevent UI overload
        return "\n".join(training_log[-20:])

    if not training_zip_upload:
        yield {
            # These keys would match Gradio output components if called directly from Gradio
            # For now, this function will be called by fluxGUI.py, which handles the UI update.
            # So, the dictionary structure is for data passing.
            "status_update": update_status("Please upload a ZIP file for training.", is_error=True),
            "generated_json_path": "",
            "final_model_path": ""
        }
        return

    yield {
        "status_update": update_status("Processing ZIP file..."),
        "generated_json_path": json_path_for_ui,
        "final_model_path": model_path_for_ui
    }

    base_extraction_dir = os.path.join(current_dir, "training_data_uploads")
    os.makedirs(base_extraction_dir, exist_ok=True)
    
    # Assuming process_zip_file is imported and accessible
    generated_json_path = process_zip_file(training_zip_upload.name, base_extraction_dir)

    if not generated_json_path:
        yield {
            "status_update": update_status("Failed to process ZIP file or metadata.jsonl not found/valid.", is_error=True),
            "generated_json_path": "",
            "final_model_path": ""
        }
        return
    
    json_path_for_ui = generated_json_path
    yield {
        "status_update": update_status(f"JSON manifest created: {json_path_for_ui}"),
        "generated_json_path": json_path_for_ui,
        "final_model_path": model_path_for_ui
    }

    results_dir_base = os.path.join(current_dir, "training_runs")
    os.makedirs(results_dir_base, exist_ok=True)
    
    # Sanitize trained_model_name_suffix if it's used directly in path
    safe_suffix = re.sub(r'[^a-zA-Z0-9_.-]', '', trained_model_name_suffix) if trained_model_name_suffix else ""
    
    run_name = f"{str(training_model_version)}_{safe_suffix if safe_suffix else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}"
    results_dir = os.path.join(results_dir_base, run_name)
    os.makedirs(results_dir, exist_ok=True)

    vae_path_prefix = os.path.join(current_dir, "audioldm2")
    config_file_path = os.path.join(current_dir, 'config/16k_64.yaml')

    yield {
        "status_update": update_status(f"Starting training run: {run_name} in {results_dir}"),
        "generated_json_path": json_path_for_ui,
        "final_model_path": model_path_for_ui
    }

    try:
        # Assuming run_training_session is imported and accessible
        for progress_update in run_training_session(
            data_path=json_path_for_ui,
            results_dir=results_dir,
            model_version=str(training_model_version), # Ensure string
            vae_path_prefix=vae_path_prefix,
            epochs=int(training_epochs),
            global_batch_size=int(training_batch_size),
            global_seed=42, 
            num_workers=int(training_num_workers),
            log_every=5, 
            accum_iter=int(training_accum_iter),
            ckpt_every=int(training_ckpt_every),
            resume_ckpt_path=None, 
            learning_rate=float(training_learning_rate),
            weight_decay=0, 
            progress_callback=None, 
            config_file_path=config_file_path
        ):
            log_message = ""
            is_error_update = False
            if progress_update.get('type') == 'log':
                log_message = f"Epoch {progress_update.get('epoch', '')} Step {progress_update.get('step', '')}: Loss: {progress_update.get('loss', ''):.4f}, Speed: {progress_update.get('steps_per_sec', ''):.2f} steps/s"
            elif progress_update.get('type') == 'info':
                log_message = progress_update.get('message', '')
            elif progress_update.get('type') == 'error':
                log_message = f"{progress_update.get('message', '')}"
                is_error_update = True
            elif progress_update.get('type') == 'model_saved':
                model_path_for_ui = progress_update.get('path', model_path_for_ui)
                log_message = f"Final model saved: {model_path_for_ui}"
            elif progress_update.get('type') == 'checkpoint_saved':
                log_message = f"Checkpoint saved: {progress_update.get('path', '')}"
            else:
                log_message = str(progress_update)
            
            yield {
                "status_update": update_status(log_message, is_error_update),
                "generated_json_path": json_path_for_ui,
                "final_model_path": model_path_for_ui
            }

        yield {
            "status_update": update_status("Training run completed."),
            "generated_json_path": json_path_for_ui,
            "final_model_path": model_path_for_ui
        }

    except Exception as e:
        import traceback
        error_message = f"An error occurred during training: {e}\n{traceback.format_exc()}"
        yield {
            "status_update": update_status(error_message, is_error=True),
            "generated_json_path": json_path_for_ui,
            "final_model_path": model_path_for_ui
        }
