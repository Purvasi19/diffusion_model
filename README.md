#  Diffusion-Based Butterfly Image Generation

##  Project Overview

This project implements a **diffusion-based image generation model** using the **Hugging Face Diffusers** library and the **Smithsonian Butterflies dataset**.

The objective is to understand the complete diffusion training pipeline, including image preprocessing, noise addition, U-Net-based noise prediction, noise scheduling, model training, image generation, loss monitoring, visualization, and model saving.

The trained model learns to generate new butterfly-like images by starting from random Gaussian noise and progressively removing the noise through an iterative denoising process.

---

##  Objectives

- Understand the working principle of diffusion models.
- Prepare and preprocess the Smithsonian Butterflies dataset.
- Implement a diffusion model using a U-Net architecture.
- Configure a DDPM noise scheduler.
- Train the model to predict added noise.
- Monitor training loss across epochs.
- Generate new butterfly images from random noise.
- Compare real and generated butterfly images.
- Analyze training metrics.
- Save and reload the trained model for inference.

---

##  How Diffusion Models Work

A diffusion model consists of two main processes:

### 1. Forward Diffusion

Noise is gradually added to a clean image.

```text
Clean Butterfly
      ↓
 Add Noise
      ↓
 More Noise
      ↓
 More Noise
      ↓
 Random Noise
