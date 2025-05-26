import torch
import os
import argparse # Will be used for a test harness later
import logging
import math
from copy import deepcopy
from torch.utils.data import DataLoader # Keep
from glob import glob
import yaml 
from collections import OrderedDict 
from time import time 
from einops import rearrange, repeat

# Assuming these are found in paths relative to where this script will be run
# or are installed packages.
from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan
# audioldm2 utilities will be called by functions defined later.
# from audioldm2.utilities.data.dataset import AudioDataset 

from constants import build_model # Assuming constants.py is accessible
from utils import load_clip, load_clap, load_t5 # Assuming utils.py is accessible

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- RF Class (Copy from original train.py) ---
class RF(torch.nn.Module):
    def __init__(self, ln=True):
        super().__init__()
        self.ln = ln
        self.stratified = False

    def forward(self, model, x, **kwargs):
        b = x.size(0)
        if self.ln:
            if self.stratified:
                quantiles = torch.linspace(0, 1, b + 1).to(x.device)
                z = quantiles[:-1] + torch.rand((b,)).to(x.device) / b
                z = torch.erfinv(2 * z - 1) * math.sqrt(2)
                t = torch.sigmoid(z)
            else:
                nt = torch.randn((b,)).to(x.device)
                t = torch.sigmoid(nt)
        else: 
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        
        zt, t = zt.to(x.dtype), t.to(x.dtype)
        vtheta = model(x=zt, t=t, **kwargs) 
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        return batchwise_mse.mean(), {} # Simplified loss return

    @torch.no_grad()
    def sample(self, model, z, conds, null_cond=None, sample_steps=50, cfg=2.0, **kwargs):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt_tensor = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z.clone()] # Clone initial z
        for i in range(sample_steps, 0, -1):
            t_val = i / sample_steps
            t = torch.full((b,), t_val, device=z.device, dtype=z.dtype) # Use torch.full for t
            vc = model(x=z, t=t, **conds)
            if null_cond is not None:
                vu = model(x=z, t=t, **null_cond)
                vc = vu + cfg * (vc - vu)
            z = z - dt_tensor * vc
            images.append(z.clone()) # Clone z for storing trajectory
        return images

    @torch.no_grad()
    def sample_with_xps(self, model, z, conds, null_cond=None, sample_steps=50, cfg=2.0, **kwargs):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt_tensor = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z.clone()] # Clone initial z
        for i in range(sample_steps, 0, -1):
            t_val = i / sample_steps
            t = torch.full((b,), t_val, device=z.device, dtype=z.dtype) # Use torch.full for t
            vc = model(x=z, t=t, **conds)
            if null_cond is not None:
                vu = model(x=z, t=t, **null_cond)
                vc = vu + cfg * (vc - vu)
            z = z - dt_tensor * vc
            images.append(z.clone()) # Append the denoised step z
        return images

# --- Helper Functions (Copied/Adapted from original train.py) ---
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if param.requires_grad: # Only update parameters that are trained
             ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    pass

def create_logger(logging_dir, is_master=True):
    # Ensure logging_dir exists if is_master
    if is_master and logging_dir: # Added check for logging_dir not None
        os.makedirs(logging_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    # Clear existing handlers to prevent duplicate logs if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    if is_master:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")] if logging_dir else [logging.StreamHandler()]
        )
    else:
        logger.addHandler(logging.NullHandler())
    return logger

# --- prepare_model_inputs (Copied from original train.py, placeholder for 'args' will be refactored later) ---
# Note: This function still uses 'args' as its first parameter. This will be addressed
# when we integrate it into run_training_session. For now, copy as is.
def prepare_model_inputs(args_or_params, batch, device, vae, clip, t5):
    # In future, args_or_params will be the dictionary of parameters for run_training_session
    text_embedding, text_embedding_mask = batch['text_embedding'], batch['text_embedding_mask']
    text_embedding_t5, text_embedding_mask_t5 = batch['text_embedding_t5'], batch['text_embedding_mask_t5']

    text_embedding = text_embedding.to(device)
    text_embedding_mask = text_embedding_mask.to(device) 
    with torch.no_grad():
        encoder_hidden_states = clip.hf_module(
            text_embedding, # .to(device) is redundant if already on device
            attention_mask=text_embedding_mask,
            output_hidden_states=False,
        )["pooler_output"]
    
    text_embedding_t5 = text_embedding_t5.to(device).squeeze(1)
    text_embedding_mask_t5 = text_embedding_mask_t5.to(device).squeeze(1)
    with torch.no_grad():
        output_t5 = t5.hf_module(
            input_ids=text_embedding_t5,
            attention_mask=text_embedding_mask_t5,
            output_hidden_states=False,
        )
        encoder_hidden_states_t5 = output_t5["last_hidden_state"].detach() 

    with torch.no_grad():
        log_mel_spec_processed = batch['log_mel_spec'].to(device)
        if log_mel_spec_processed.ndim == 3: # Ensure 4D: (B, C, H, W)
            log_mel_spec_processed = log_mel_spec_processed.unsqueeze(1)
        image = vae.encode(log_mel_spec_processed).latent_dist.sample().mul_(vae.config.scaling_factor) 

    bs, c, h, w = image.shape
    image = rearrange(image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).float()
    
    # Positional embedding logic from original train.py
    img_ids = torch.zeros(h // 2, w // 2, 3, device=device) # Ensure on correct device
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt_ids = torch.zeros(bs, encoder_hidden_states_t5.shape[1], 3, device=device) # Ensure on correct device
    
    model_kwargs = dict(
        img_ids=img_ids.to(image.device), # Redundant .to(image.device) if already on device
        txt=encoder_hidden_states_t5.to(image.device).float(),
        txt_ids=txt_ids.to(image.device),
        y=encoder_hidden_states.to(image.device).float(),
    )
    return image, model_kwargs

# --- run_training_session Function Definition ---
# (This function will be appended to train_refactored.py)

# Import AudioDataset here, as it's used within this function
from audioldm2.utilities.data.dataset import AudioDataset

def run_training_session(
    data_path: str,
    results_dir: str,
    model_version: str,
    vae_path_prefix: str, # e.g., './audioldm2' or specific path to 'vae' parent
    epochs: int,
    global_batch_size: int,
    global_seed: int = 1234,
    num_workers: int = 4,
    log_every: int = 100,
    accum_iter: int = 16,
    ckpt_every: int = 100000,
    resume_ckpt_path: str = None,
    learning_rate: float = 3e-5,
    weight_decay: float = 0,
    progress_callback: callable = None,
    config_file_path: str = 'config/16k_64.yaml' # Relative to project root
):
    # 1. Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    if progress_callback: progress_callback({'type': 'info', 'message': f"Using device: {device}"})
    else: print(f"Using device: {device}")

    # 2. Seed initialization
    torch.manual_seed(global_seed)

    # 3. Setup experiment folder & logger
    os.makedirs(results_dir, exist_ok=True) 
    model_string_name = model_version.replace("/", "-")
    # Ensure experiment_dir is unique if results_dir might contain other experiments
    existing_experiments = glob(f"{results_dir}/{model_string_name}_*")
    experiment_num = len(existing_experiments)
    experiment_dir = f"{results_dir}/{model_string_name}_{experiment_num}"

    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True) # Ensure checkpoint_dir is also created

    logger = create_logger(experiment_dir, is_master=True)
    logger.info(f"Experiment directory created at {experiment_dir}")
    if progress_callback: progress_callback({'type': 'info', 'message': f"Experiment dir: {experiment_dir}"})

    # 4. Build Model
    model = build_model(model_version).to(device)
    if progress_callback: progress_callback({'type': 'info', 'message': f"Model version {model_version} built."})
    
    parameters_sum = sum(x.numel() for x in model.parameters())
    logger.info(f"Model Parameters: {parameters_sum / 1000000.0:.2f} M")
    if progress_callback: progress_callback({'type': 'info', 'message': f"Model Parameters: {parameters_sum / 1000000.0:.2f} M"})

    # Initialize ckpt_data to None
    ckpt_data = None
    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        logger.info(f"Loading checkpoint from: {resume_ckpt_path}")
        if progress_callback: progress_callback({'type': 'info', 'message': f"Loading checkpoint: {resume_ckpt_path}"})
        try:
            ckpt_data = torch.load(resume_ckpt_path, map_location=device)
            if 'ema' in ckpt_data:
                state_dict_to_load = ckpt_data['ema']
            elif 'model' in ckpt_data: 
                state_dict_to_load = ckpt_data['model']
            else:
                state_dict_to_load = ckpt_data 

            model_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict_to_load.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
            model.load_state_dict(filtered_state_dict, strict=False) 
            logger.info("Checkpoint loaded successfully.")
            if progress_callback: progress_callback({'type': 'info', 'message': "Checkpoint loaded."})
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            if progress_callback: progress_callback({'type': 'error', 'message': f"Error loading checkpoint: {e}"})

    # 5. EMA Model
    ema = deepcopy(model).to(device)
    requires_grad(ema, False) 
    if progress_callback: progress_callback({'type': 'info', 'message': "EMA model created/synced."})
    
    update_ema(ema, model, decay=0) 

    # 6. Load VAE, T5, CLAP
    actual_vae_path = vae_path_prefix
    if not os.path.basename(actual_vae_path) == 'vae': 
         actual_vae_path = os.path.join(vae_path_prefix, 'vae')

    if not os.path.isdir(actual_vae_path):
        logger.error(f"VAE path not found or not a directory: {actual_vae_path}. Expected structure: .../audioldm2/vae or path directly to 'vae' folder.")
        if progress_callback: progress_callback({'type': 'error', 'message': f"VAE path error: {actual_vae_path}"})
        return

    try:
        vae = AutoencoderKL.from_pretrained(actual_vae_path).to(device)
        if progress_callback: progress_callback({'type': 'info', 'message': f"VAE loaded from {actual_vae_path}."})
    except Exception as e:
        logger.error(f"Error loading VAE from {actual_vae_path}: {e}")
        if progress_callback: progress_callback({'type': 'error', 'message': f"Error loading VAE: {e}"})
        return

    try:
        t5 = load_t5(device, max_length=256)
        clap = load_clap(device, max_length=256)
        if progress_callback: progress_callback({'type': 'info', 'message': "T5 and CLAP models loaded."})
    except Exception as e:
        logger.error(f"Error loading T5/CLAP: {e}")
        if progress_callback: progress_callback({'type': 'error', 'message': f"Error loading T5/CLAP: {e}"})
        return

    # 7. Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if progress_callback: progress_callback({'type': 'info', 'message': "Optimizer created."})
    
    if ckpt_data and 'opt' in ckpt_data: # Check if ckpt_data was loaded and has 'opt'
        try:
            opt.load_state_dict(ckpt_data['opt'])
            logger.info("Optimizer state loaded from checkpoint.")
            if progress_callback: progress_callback({'type': 'info', 'message': "Optimizer state loaded."})
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")
            if progress_callback: progress_callback({'type': 'warning', 'message': f"Could not load optimizer state: {e}"})

    # 8. Diffusion
    diffusion = RF()
    if progress_callback: progress_callback({'type': 'info', 'message': "Diffusion RF object created."})
    
    # --- Dataset and DataLoader ---
    logger.info(f"Loading dataset config from: {config_file_path}")
    if not os.path.exists(config_file_path):
        logger.error(f"Dataset config file not found: {config_file_path}")
        if progress_callback: progress_callback({'type': 'error', 'message': f"Dataset config missing: {config_file_path}"})
        return
    
    try:
        with open(config_file_path, 'r') as f:
            dataset_config = yaml.safe_load(f) # Use safe_load
    except Exception as e:
        logger.error(f"Error loading dataset config YAML: {e}")
        if progress_callback: progress_callback({'type': 'error', 'message': f"Dataset config error: {e}"})
        return

    logger.info(f"Initializing dataset from JSON: {data_path}")
    if not os.path.exists(data_path):
        logger.error(f"Training data JSON file not found: {data_path}")
        if progress_callback: progress_callback({'type': 'error', 'message': f"Training data JSON missing: {data_path}"})
        return

    dataset = AudioDataset(
        config=dataset_config, # Use loaded config
        split="train", 
        waveform_only=False, 
        dataset_json_path=data_path, 
        tokenizer=clap.tokenizer, 
        uncond_pro=0.1, # These could be params later
        text_ctx_len=77, # These could be params later
        tokenizer_t5=t5.tokenizer,
        text_ctx_len_t5=256, # These could be params later
        uncond_pro_t5=0.1 # These could be params later
    )
    logger.info(f"Dataset contains {len(dataset):,} items.")
    if progress_callback: progress_callback({'type': 'info', 'message': f"Dataset contains {len(dataset):,} items."})

    loader = DataLoader(
        dataset,
        batch_size=global_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    if progress_callback: progress_callback({'type': 'info', 'message': "DataLoader created."})

    # --- Training Loop ---
    model.train() # Set model to train mode
    ema.eval()    # EMA model is always in eval mode

    train_steps = 0
    # Resume train_steps if available in checkpoint args
    if ckpt_data and 'args' in ckpt_data and 'train_steps' in ckpt_data['args']:
        train_steps = ckpt_data['args']['train_steps']
        logger.info(f"Resuming train_steps from checkpoint: {train_steps}")

    log_steps = 0
    running_loss = 0
    start_time = time() 

    logger.info(f"Training for {epochs} epochs...")
    if progress_callback: progress_callback({'type': 'training_start', 'epochs': epochs})

    for epoch in range(epochs):
        logger.info(f"Beginning epoch {epoch+1}/{epochs}...")
        if progress_callback: progress_callback({'type': 'epoch_start', 'epoch': epoch + 1, 'total_epochs': epochs})
        
        data_iter_step = 0
        for batch_idx, batch in enumerate(loader):
            # For prepare_model_inputs, we pass a dictionary of relevant params instead of 'args'
            # This needs prepare_model_inputs to be adapted or use a simple namespace/dict for its first arg.
            # For now, we assume prepare_model_inputs can handle this or we prepare a temp object.
            # Let's create a temporary params dict for prepare_model_inputs for now
            # This is a simplification; prepare_model_inputs might need more specific args.
            current_params_for_prepare = {} # Empty for now, as prepare_model_inputs doesn't use args for its logic

            latents, model_kwargs = prepare_model_inputs(current_params_for_prepare, batch, device, vae, clap, t5)
            
            loss, _ = diffusion.forward(model=model, x=latents, **model_kwargs) 
            
            # Gradient accumulation
            loss = loss / accum_iter 
            loss.backward()

            if (data_iter_step + 1) % accum_iter == 0:
                opt.step()
                opt.zero_grad()
                update_ema(ema, model) # Update EMA model

            data_iter_step += 1
            running_loss += loss.item() * accum_iter # Scale loss back for logging
            log_steps += 1
            train_steps += 1

            if train_steps % log_every == 0:
                if device == "cuda": torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time) if (end_time - start_time) > 0 else 0.0
                avg_loss = running_loss / log_steps
                
                log_msg = f"(Epoch {epoch+1}, Step {train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}"
                logger.info(log_msg)
                if progress_callback: progress_callback({'type': 'log', 'epoch': epoch+1, 'step': train_steps, 'loss': avg_loss, 'steps_per_sec': steps_per_sec})
                
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    # Store key training parameters for reference
                    "args": {
                        "model_version": model_version, "epochs_completed": epoch + 1, 
                        "global_batch_size": global_batch_size, "learning_rate": learning_rate,
                        "train_steps": train_steps
                    }
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}_ema.pt"
                try:
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved EMA checkpoint to {checkpoint_path}")
                    if progress_callback: progress_callback({'type': 'checkpoint_saved', 'path': checkpoint_path})
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")
                    if progress_callback: progress_callback({'type': 'error', 'message': f"Error saving checkpoint: {e}"})
        
    logger.info("Training finished.")
    if progress_callback: progress_callback({'type': 'training_end', 'message': "Training finished."})
    
    # Save final model
    final_model_path = f"{experiment_dir}/final_ema_model.pt"
    final_checkpoint = {"ema": ema.state_dict(), "args": {"model_version": model_version, "epochs_trained":epochs, "final_train_steps": train_steps}}
    try:
        torch.save(final_checkpoint, final_model_path)
        logger.info(f"Saved final EMA model to {final_model_path}")
        if progress_callback: progress_callback({'type': 'model_saved', 'path': final_model_path})
        return final_model_path
    except Exception as e:
        logger.error(f"Error saving final model: {e}")
        if progress_callback: progress_callback({'type': 'error', 'message': f"Error saving final model: {e}"})
        return None

# --- End of run_training_session Function Definition ---

if __name__ == "__main__":
    print("Starting a test run of run_training_session...")

    # Create dummy data and config for testing if they don't exist
    # This is a simplified setup. Real training requires actual data.
    
    # Dummy JSON data path (e.g., for one audio file)
    # User should replace this with a real JSON path for actual testing.
    test_data_json_path = "dummy_train_data.json"
    # Dummy config path
    test_config_yaml_path = "dummy_config.yaml"

    # Create results directory for test run
    test_results_dir = "test_training_results"
    os.makedirs(test_results_dir, exist_ok=True)

    # Create dummy VAE directory structure if needed for from_pretrained to not fail immediately
    # This is a hack for basic script execution testing, not for functional VAE loading.
    # Real VAE models are needed for actual training.
    dummy_vae_path = os.path.join(test_results_dir, "dummy_audioldm2_vae")
    # os.makedirs(dummy_vae_path, exist_ok=True) # No, vae_path_prefix should point to parent of 'vae'
    dummy_audioldm2_root = os.path.join(test_results_dir, "dummy_audioldm2") # Parent
    dummy_vae_subfolder = os.path.join(dummy_audioldm2_root, "vae") # Actual 'vae' subfolder
    os.makedirs(dummy_vae_subfolder, exist_ok=True)
    # Create a dummy model file or config.json so from_pretrained doesn't immediately fail
    # This won't make VAE work, but helps test script flow.
    try:
        with open(os.path.join(dummy_vae_subfolder, "config.json"), "w") as f:
            f.write('{"_class_name": "AutoencoderKL", "_diffusers_version": "0.20.0"}') # Minimal config
    except Exception as e:
        print(f"Could not create dummy VAE config: {e}")


    # Create dummy training data JSON if it doesn't exist
    if not os.path.exists(test_data_json_path):
        try:
            with open(test_data_json_path, "w") as f:
                # Example: one dummy audio entry. Actual file paths would be needed.
                # Duration, mel_length also need to be accurate for real data.
                # This also assumes 'dummy.wav' and 'dummy.npy' exist or are handled by AudioDataset.
                # For a simple script execution test, AudioDataset might not even be fully loaded.
                f.write('[{"text": "dummy prompt", "duration": 10.0, "mel_length": 512, "path": "dummy.wav", "text_emb_path": "dummy_text_emb.npy", "t5_text_emb_path": "dummy_t5_emb.npy"}]')
            print(f"Created dummy data JSON: {test_data_json_path}")
        except Exception as e:
            print(f"Could not create dummy data JSON: {e}")

    # Create dummy dataset config YAML if it doesn't exist
    if not os.path.exists(test_config_yaml_path):
        try:
            with open(test_config_yaml_path, "w") as f:
                f.write("dataset:\n  train:\n    - path/to/dummy/data\n  test:\n    - path/to/dummy/data\n") # Minimal structure
            print(f"Created dummy config YAML: {test_config_yaml_path}")
        except Exception as e:
            print(f"Could not create dummy config YAML: {e}")
    
    # Define a simple progress callback for testing
    def test_progress_callback(update):
        print(f"Progress Update: {update}")

    print(f"Attempting to run training with potentially dummy/incomplete data and VAE.")
    print(f"This test primarily checks script execution flow, not full training functionality.")
    print(f"Ensure actual data, config, and VAE models are provided for real training.")

    try:
        final_model_path = run_training_session(
            data_path=test_data_json_path, # Use dummy path
            results_dir=test_results_dir,
            model_version="small",  # Or any valid small model version string from constants.py
            vae_path_prefix=dummy_audioldm2_root, # Path to parent of 'vae' folder
            epochs=1, # Minimal epochs for testing
            global_batch_size=1, # Minimal batch size
            global_seed=42,
            num_workers=0, # Often 0 for easier debugging
            log_every=1,
            accum_iter=1,
            ckpt_every=1, # Checkpoint quickly for test
            resume_ckpt_path=None,
            learning_rate=1e-5,
            progress_callback=test_progress_callback,
            config_file_path=test_config_yaml_path # Use dummy config
        )
        if final_model_path:
            print(f"Test run completed. Final model supposedly at: {final_model_path}")
        else:
            print("Test run completed, but final model path was not returned (possibly due to an error during training setup).")
    except Exception as e:
        print(f"Error during test run of run_training_session: {e}")
        import traceback
        traceback.print_exc()
