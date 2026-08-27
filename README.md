# Diffusion-Based Butterfly Image Generation

## Project Overview

This project implements a **diffusion-based image generation model** using the **Hugging Face Diffusers** library and the **Smithsonian Butterflies dataset**.

The objective is to understand the complete diffusion training pipeline, including image preprocessing, noise addition, U-Net-based noise prediction, noise scheduling, model training, image generation, loss monitoring, visualization, and model saving.

The trained model learns to generate new butterfly-like images by starting from random Gaussian noise and progressively removing the noise through an iterative denoising process.

---

## Objectives

* Understand the working principle of diffusion models.
* Prepare and preprocess the Smithsonian Butterflies dataset.
* Implement a diffusion model using a U-Net architecture.
* Configure a DDPM noise scheduler.
* Train the model to predict added noise.
* Monitor training loss across epochs.
* Generate new butterfly images from random noise.
* Compare real and generated butterfly images.
* Analyze training metrics.
* Save and reload the trained model for inference.

---

## How Diffusion Models Work

A diffusion model consists of two main processes:

### 1. Forward Diffusion

Noise is gradually added to a clean image until the image becomes almost completely random noise.

```text
Clean Butterfly
      ↓
   Add Noise
      ↓
   More Noise
      ↓
   More Noise
      ↓
 Random Gaussian Noise
```

The forward process can be represented as:

```text
x₀ → x₁ → x₂ → ... → xₜ
```

where:

* `x₀` = original clean image
* `xₜ` = noisy image at timestep `t`
* `t` = diffusion timestep

The model uses a **DDPM (Denoising Diffusion Probabilistic Model) scheduler** to control how noise is added at each timestep.

### 2. Reverse Diffusion

The reverse process starts with random Gaussian noise and gradually removes the noise to generate an image.

```text
Random Noise
      ↓
  Denoising
      ↓
  Denoising
      ↓
  Denoising
      ↓
Generated Butterfly
```

The **U-Net** predicts the noise present in the image at each timestep. The scheduler then uses this prediction to obtain a slightly less noisy image.

```text
Random Noise
     ↓
   U-Net
     ↓
Predict Noise
     ↓
Scheduler
     ↓
Less Noisy Image
     ↓
Repeat
     ↓
Generated Image
```

---

## Model Architecture

The diffusion model uses a **U-Net architecture** as the noise prediction network.

The main components are:

### U-Net

The U-Net receives:

* A noisy image
* The current diffusion timestep

and predicts the noise added to the image.

```text
              Noisy Image
                   │
                   ▼
             ┌───────────┐
             │  Encoder  │
             └─────┬─────┘
                   ↓
              Bottleneck
                   ↓
             ┌─────┴─────┐
             │  Decoder  │
             └─────┬─────┘
                   ↓
             Predicted Noise
```

The encoder extracts image features while progressively reducing the spatial dimensions. The decoder reconstructs the spatial representation using the learned features.

Skip connections between the encoder and decoder help preserve important image details.

---

## Gaussian Noise

The model uses **Gaussian noise** during the forward diffusion process.

Gaussian noise is random noise whose values follow a normal distribution.

It is commonly used in diffusion models because it is mathematically convenient and allows the gradual corruption and reconstruction process to be formulated efficiently.

The noise is added according to the noise schedule controlled by the DDPM scheduler.

---

## Diffusion Pipeline

The complete training pipeline is:

```text
Smithsonian Butterflies Dataset
              ↓
       Image Preprocessing
              ↓
        Resize Images
              ↓
       Normalize Images
              ↓
       Sample Timestep
              ↓
       Add Gaussian Noise
              ↓
        Noisy Image
              ↓
           U-Net
              ↓
      Predict Added Noise
              ↓
     Calculate MSE Loss
              ↓
      Backpropagation
              ↓
        Update Weights
              ↓
          Repeat
```

---

##  Dataset

The project uses the **Smithsonian Butterflies dataset**, containing butterfly images that are used to train the diffusion model.

### Dataset Preparation

The images are processed before being passed to the model.

The preprocessing pipeline includes:

1. Loading the butterfly images.
2. Resizing images to a consistent resolution.
3. Converting images into tensors.
4. Normalizing pixel values.
5. Creating batches using a DataLoader.

Consistent image dimensions are required because the neural network expects tensors with the same spatial dimensions within a batch.

---

## Training Process

During training, the model learns to predict the noise that was added to an image.

For every training iteration:

1. A clean butterfly image is selected.
2. A random diffusion timestep is selected.
3. Gaussian noise is generated.
4. Noise is added to the clean image.
5. The noisy image and timestep are passed to the U-Net.
6. The U-Net predicts the noise.
7. The predicted noise is compared with the actual noise.
8. Mean Squared Error (MSE) is calculated.
9. Backpropagation is performed.
10. Model parameters are updated using the optimizer.

### Loss Function

The training objective is to minimize the difference between the actual noise and the noise predicted by the U-Net.

```text
Loss = MSE(Actual Noise, Predicted Noise)
```

A lower loss indicates that the model is becoming better at predicting the noise.

---

##  Training Results

The training loss was monitored across epochs to evaluate the learning process.

### Observed Metrics

| Metric                 |   Value |
| ---------------------- | ------: |
| Initial Loss           | 0.03116 |
| Final Loss             | 0.02729 |
| Minimum Loss           | 0.02493 |
| Best Epoch             |       9 |
| Overall Loss Reduction |  12.44% |

The overall decrease in loss indicates that the model learned to predict the noise added during the diffusion process.

The minimum loss was observed at **Epoch 9**, indicating the best training performance in the experiment.

---

## Image Generation

After training, the model can generate new butterfly images.

Generation begins with random Gaussian noise:

```text
Random Gaussian Noise
          ↓
       U-Net
          ↓
     Noise Prediction
          ↓
       Scheduler
          ↓
     Denoising Step
          ↓
       Repeat
          ↓
   Generated Butterfly
```

The denoising process is repeated for multiple timesteps until the random noise is transformed into a meaningful image.

---

## Real vs Generated Images

The generated images can be visually compared with real butterfly images from the dataset.

The comparison helps evaluate:

* Overall butterfly-like structure
* Shape and appearance
* Color patterns
* Image quality
* Diversity of generated samples

> Note: Since this is an experimental diffusion model, generated images may not perfectly reproduce realistic butterflies. The primary goal of this project is to understand and implement the diffusion training pipeline.

---

## Model Saving and Loading

After training, the trained model and scheduler can be saved for later use.

```text
Training
   ↓
Trained U-Net
   ↓
Save Model
   ↓
Load Model
   ↓
Inference
   ↓
Generate Images
```

Saving the model allows image generation without having to retrain the network from scratch.

---

## Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **Hugging Face Diffusers**
* **NumPy**
* **Matplotlib**
* **Smithsonian Butterflies Dataset**
* **DDPM Scheduler**
* **U-Net**

---
## Conclusion

This project demonstrates the complete workflow of a diffusion-based image generation system using butterfly images.

The model learns the relationship between noisy and clean images by predicting the noise introduced during the forward diffusion process. During inference, the learned noise prediction capability is used in the reverse diffusion process to gradually transform random Gaussian noise into a new butterfly-like image.

The experiment provides a practical understanding of how **U-Net, noise scheduling, forward diffusion, reverse diffusion, and iterative denoising** work together to perform image generation.

---

## 🔮 Future Improvements

Possible improvements include:

* Training for more epochs.
* Using a larger butterfly dataset.
* Increasing image resolution.
* Hyperparameter tuning.
* Experimenting with different noise schedules.
* Using a larger U-Net architecture.
* Generating more diverse samples.
* Evaluating generated images using quantitative metrics.
* Experimenting with conditional diffusion for controlled butterfly generation.
---

