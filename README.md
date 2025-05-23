# Anime-Characterization-

    ## 🧪 Methodology

    ### 1. Problem Definition
    The goal of this phase is to classify facial images as either:

    - **Real human faces** (from the CelebA dataset), or  
    - **AI-generated human faces** (from the FFHQ dataset),  

    as a foundational step toward anime character behavior analysis.

    ---

    ### 2. Data Collection and Preprocessing
    - **Real Faces**: We used the CelebA dataset which includes:
        - Facial attribute annotations  
        - Bounding boxes  
        - Landmark points  
        - Evaluation partitions  

    - **AI Faces**: We used a processed version of the FFHQ dataset containing GAN-generated human-like faces.

    - **Data Labeling**:
        - `label = 0` for real images  
        - `label = 1` for AI-generated images  

    - **Preprocessing**:
        - Resizing to `64×64`  
        - Normalization using `ToTensor()` from `torchvision.transforms`  

    ---

    ### 3. Dataset Construction
    We implemented a custom PyTorch `Dataset` class to:
    - Dynamically load images from respective folders  
    - Apply the transformation pipeline  
    - Return images and labels in the required format  

    The full dataset was then split using `train_test_split()`:
    - **80%** for training  
    - **20%** for validation  

    ---

    ### 4. Model Architecture
    We developed a **Simple Convolutional Neural Network (CNN)** with the following layers:
    - `Conv2D → ReLU → MaxPool` × 2  
    - Fully connected layer (`fc1`) with ReLU  
    - Output layer (`fc2`) with sigmoid activation (for binary classification)  

    **Training Details**:
    - Loss Function: Binary Cross Entropy Loss  
    - Optimizer: Adam Optimizer with `lr = 0.001`  
    - GPU acceleration using PyTorch’s `cuda` (if available)  

    ---

    ### 5. Training and Validation
    The model was trained for **15 epochs**. After each epoch, we recorded:
    - Training loss  
    - Validation loss  
    - Validation accuracy  

    Visualizations of loss and accuracy trends were plotted using Matplotlib.

    ---

    ### 6. Model Saving and Inference
    - The trained model weights were saved using `torch.save()`.  
    - An inference function was implemented to classify any input image as real or AI-generated:

    ```python
    if prediction.item() == 0:
            print("Real Human Image")
    else:
            print("AI Generated Image")
    ```

    ---

    ### 7. Future Work (Persistent Homology Integration)
    As part of our collaboration, we plan to enhance the model by integrating **Topological Data Analysis (TDA)** using Persistent Homology:
    - To extract shape and connectivity features from face images  
    - Combine them with CNN features for improved classification accuracy and robustness  
