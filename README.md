# Deep Convolutional Generative Adversarial Network (DCGAN)

## Dataset Preprocessing Steps
The model is trained on the **CelebA Faces** dataset, which consists of large-scale high-resolution images of human faces. Since the dataset does not have labels, we only preprocess images before feeding them into the model.

### Steps:
1. **Download the Dataset**
   - The dataset was downloaded from Kaggle as a `.zip` archive.
   - Extracted the contents into a local directory.

2. **Loading Images**
   - Used `torchvision.datasets.ImageFolder` to load images from the extracted folder.
   - Applied transformations using `torchvision.transforms`:
     - Resized images to **64x64** pixels.
     - Converted images to **tensors**.
     - Normalized pixel values to **[-1, 1]** for stable training.

3. **Creating DataLoader**
   - Used `torch.utils.data.DataLoader` to batch and shuffle images.
   - Set `batch_size = 100` for efficient training.

---

## How to Train and Test the Model
### Training Process:
1. **Model Architecture**
   - **Generator (G):**
     - Converts random noise into realistic images.
     - Uses transposed convolutions and batch normalization.
   - **Discriminator (D):**
     - Classifies real and generated images.
     - Uses standard convolutional layers with Leaky ReLU activation.

2. **Training Loop**
   - **Discriminator Training:**
     - Processes real images and computes loss.
     - Generates fake images and computes loss.
     - Updates the Discriminator.
   - **Generator Training:**
     - Generates fake images and computes loss.
     - Updates the Generator.
   - **Saving Generated Images:**
     - At each epoch, generated images are saved to track progress.

3. **Loss Function & Optimizer**
   - **Binary Cross-Entropy Loss (BCE Loss)** is used.
   - **Adam Optimizer** is applied to both networks.

4. **Running Training**
   ```bash
   python train.py  # If using a Python script
   ```
   - The model trains for multiple epochs, saving progress periodically.

### Testing / Generating Images
- After training, use the trained Generator to create new images:
  ```python
  noise = torch.randn(1, 100, 1, 1, device=device)
  generated_image = netG(noise)
  ```
- Save generated images using:
  ```python
  torchvision.utils.save_image(generated_image, "output.png", normalize=True)
  ```

---

## Expected Outputs
### Training Observations:
- **Discriminator Performance:**
  - LossD starts around **0.53** and gradually improves.
  - Stabilizes around **0.43** as it learns to distinguish real vs. fake images.
  
- **Generator Performance:**
  - LossG starts high (~2.2) and improves over time.
  - Peaks around **3.8** but might fluctuate slightly.

### Generated Image Quality:
- At the beginning, generated images look **noisy and unrealistic**.
- After several epochs, images resemble **human faces** but might still lack sharpness.
- With extended training, the Generator produces **clearer and more defined facial features**.

---

## Notes
- Ensure `dataloader` correctly loads images as Tensors before training.
- If encountering `'list' object has no attribute 'to'` error, verify that `dataloader` outputs correctly formatted tensors.
- Training on CPU may be slow; use GPU if available (`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).

