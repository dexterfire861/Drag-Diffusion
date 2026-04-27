# Drag Diffusion

Texture-preserving object relocation using a Stable Diffusion 2.1 pipeline with DDPM inversion and noise-prior shifting.

This project lets you upload an image, mark the object you want to move, mark the target location, and generate a harmonized result. It includes a Gradio interface, a Colab-style demo script, a quick synthetic test, and evaluation utilities for comparing the noise-shift method against a baseline SDEdit run.

## Features

- Interactive Gradio app for object relocation.
- Source and target mask painting directly in the browser.
- DDPM inversion plus noise-shift relocation for texture preservation.
- Baseline toggle using SDEdit without noise shifting.
- Perceptual-distance scoring and visualization helpers.
- Quick self-contained test scene for smoke testing.

## Project Structure

```text
.
├── app.py                         # Gradio UI
├── main.py                        # Small MPS availability check
├── quick_test.py                  # Synthetic end-to-end test
├── run_eval.py                    # Batch evaluation script
├── colab_demo.py                  # Colab-oriented walkthrough
├── pipeline/
│   └── relocation_pipeline.py     # Main object relocation pipeline
├── utils/                         # Image and mask utilities
├── eval/                          # Metrics and visualization utilities
└── requirements.txt               # Python dependencies
```

## Requirements

- Python 3.10 or newer recommended.
- A CUDA GPU is recommended for practical runtime.
- Apple Silicon MPS is supported, but the pipeline uses smaller 512px images to reduce memory pressure.
- The first model download requires access to `stabilityai/stable-diffusion-2-1` from Hugging Face.

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download/cache the Stable Diffusion components:

```bash
python test_setup.py
```

If Hugging Face access is required, log in first:

```bash
huggingface-cli login
```

## Run the App

Start the Gradio interface:

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

Workflow:

1. Upload an image.
2. Paint the source mask over the object to move.
3. Paint the target mask where the object should move.
4. Enter a prompt describing the final scene.
5. Click **Run**.

Use the **DDPM noise shift** checkbox to compare the main method against the baseline.

## Quick Test

Run a synthetic end-to-end test:

```bash
python quick_test.py
```

Outputs are written to:

```text
data/results/
```

The main comparison image is:

```text
data/results/quick_test.png
```

## Batch Evaluation

`run_eval.py` expects test cases in this format:

```text
data/test_images/<case_name>/
├── image.jpg or image.png
├── src_mask.png
├── tgt_mask.png
└── prompt.txt
```

Run evaluation with:

```bash
python run_eval.py
```

Results and comparison figures are saved under:

```text
data/results/
```

## Colab

See `colab_demo.py` and `Drag_Diffusion_Colab.ipynb` for a notebook-style workflow. In Colab, use a GPU runtime and allow the model to download on the first run.

## Notes

- Lower perceptual distance generally means better object texture preservation.
- Lower SDEdit strength keeps the copy-paste result more faithful.
- Higher SDEdit strength can improve harmonization, but may alter object texture.
- Runtime depends heavily on GPU memory, inference steps, and image size.
