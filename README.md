## FluxMusic: Text-to-Music Generation with Rectified Flow Transformer <br><sub>GUI Implementation</sub>

<a href="https://arxiv.org/abs/2409.00587"><img src="https://img.shields.io/static/v1?label=Paper&message=FluxMusic&color=purple&logo=arxiv"></a> &ensp;
<a href="https://huggingface.co/feizhengcong/fluxmusic"><img src="https://img.shields.io/static/v1?label=Models&message=HuggingFace&color=yellow"></a> &ensp;

This repo contains a Graphical User Interface (GUI) implementation of the FluxMusic model, based on the paper *Flux that plays music*. It explores a simple extension of diffusion-based rectified flow Transformers for text-to-music generation.

### FluxMusic GUI

We have created a user-friendly GUI for FluxMusic using Gradio. This interface allows users to easily generate music based on text prompts without needing to interact with command-line interfaces.

#### Features:

1. **Model Selection**: Users can choose from different FluxMusic models (small, base, large, giant) via a dropdown menu.

2. **Text Prompt**: Enter your desired text prompt to guide the music generation.

3. **Sliders and Inputs**:
   - **Seed**: Set a seed for reproducibility (0 for random).
   - **CFG Scale**: Adjust the Classifier-Free Guidance scale (1-40).
   - **Steps**: Set the number of diffusion steps (10-200).
   - **Duration**: Specify the length of the generated audio in seconds (10-300).

4. **File Management**:
   - **Models Folder**: Place your FluxMusic model files (`.pt`) in the `models` folder.
   - **Generations Folder**: Generated audio files are saved in the `generations` folder.

5. **File Naming System**: Generated files are named using the format: `[prompt]_[seed]_[model]_[counter].wav`

### Setup and Running

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Place your FluxMusic model files in the `models` folder.

3. Run the GUI:
   ```
   python fluxGUI.py
   ```

4. Use the interface to generate music based on your prompts and preferences.

### Model Information

FluxMusic comes in four sizes: Small, Base, Large, and Giant. You can download these models from the following links:

|  Model | Url |
|---------------|------------------|
| FluxMusic-Small | [link](https://huggingface.co/feizhengcong/FluxMusic/blob/main/musicflow_s.pt) |
| FluxMusic-Base  | [link](https://huggingface.co/feizhengcong/FluxMusic/blob/main/musicflow_b.pt) |
| FluxMusic-Large | [link](https://huggingface.co/feizhengcong/FluxMusic/blob/main/musicflow_l.pt) |
| FluxMusic-Giant | [link](https://huggingface.co/feizhengcong/FluxMusic/blob/main/musicflow_g.pt) |

### Acknowledgments

The codebase is based on the awesome [Flux](https://github.com/black-forest-labs/flux) and [AudioLDM2](https://github.com/haoheliu/AudioLDM2) repos.