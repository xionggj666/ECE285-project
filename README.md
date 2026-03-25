# Controllable Tarot Card Generation Pipeline

This project is an **AIGC-oriented tarot card generation pipeline** designed to improve controllability, semantic consistency, and deck-level style coherence for full tarot deck generation.

Instead of relying on prompt-only image generation, the pipeline decomposes tarot card synthesis into multiple stages: **semantic specification generation, layout guidance, diffusion rendering, LoRA-based style adaptation, and symbol-level composition**. The project has also been extended with **vLLM-based semantic planning** and **inference benchmarking**, making it closer to a practical AIGC engineering workflow.

## Project Goals

The project aims to address several common issues in prompt-only image generation:

- Maintain **consistent visual style** across an entire tarot deck
- Improve **semantic controllability** for card identity, figures, backgrounds, and symbolic elements
- Improve **count control** for repeated symbols in Minor Arcana cards
- Extend the project from a research-style prototype into a more **engineering-oriented AIGC pipeline**

## Pipeline Overview

The current pipeline consists of the following stages:

1. **Semantic Specification Generation**  
   Given a global deck prompt and a specific card name, the system generates a structured card specification (`spec`) containing:
   - card name
   - main figure
   - background
   - symbols
   - mood
   - color hints
   - optional count hints

2. **Layout Guidance**  
   The specification is converted into layout constraints or control inputs, which guide downstream rendering and symbol placement.

3. **Diffusion Rendering**  
   A diffusion-based renderer generates the card image from the deck prompt and card-specific information.  
   The rendering stage can optionally include:
   - **ControlNet** for layout and structure guidance
   - **LoRA** for deck-level style adaptation

4. **Symbol-Level Composition**  
   For Minor Arcana cards, symbol prototypes and slot-based composition are used to improve repeated-symbol placement and count accuracy.

5. **Final Card Output**  
   The rendered image and symbol composition results are combined into the final tarot card.

## Engineering Extensions

To make the project more aligned with real-world AIGC engineering workflows, the following extensions were added:

### 1. Inference Benchmarking
The project includes single-card inference benchmarking under multiple settings:

- `base`
- `LoRA-only`
- `ControlNet + LoRA`

Measured metrics include:

- runtime per image
- peak GPU memory
- overhead introduced by different inference components

### 2. LoRA / ControlNet Inference Comparison
The project compares different inference settings under the same card prompt to analyze:

- style enhancement from LoRA
- structural guidance from ControlNet
- runtime and memory overhead

### 3. vLLM-Based Semantic Planning
The original rule-based semantic parser has been extended with a **vLLM-based spec generation module** using an instruction-tuned open-source model.  
This enables the pipeline to generate structured tarot specifications from natural-language deck prompts and card identities.

The target workflow is:

`deck prompt + card name -> vLLM semantic planner -> structured spec -> layout / rendering pipeline -> final tarot card`

## Repository Structure

```text
data/
├── images/                  # image data
├── layouts/                 # layout/control images
├── layouts_new/             # updated layout/control images
├── tarot-images.json        # image metadata
├── tarot_specs.json         # tarot specification data
└── train.jsonl              # training / fine-tuning data

loras/
└── ...                      # LoRA weights

scripts/
├── serve_vllm.sh            # local/remote vLLM launch script
├── benchmark_tarot_infer.py # inference benchmark script
├── pipeline.py              # main generation pipeline
├── prepare_images.py        # data preprocessing
└── tarot_controlnet_generate.py  # ControlNet-based generation script



