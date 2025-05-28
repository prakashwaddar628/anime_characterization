# README.md

# Anime Character Recognition and Behavior Analysis

This project uses a Convolutional Neural Network (CNN) to recognize anime characters from images and lay the foundation for behavioral analysis based on character identity.

## 📌 Project Structure
```
├── datasets
│   └── processed
│       └── dataset.npz
├── models
│   └── best_anime_model.h5
├── notebooks
│   └── anime_character_recognition.ipynb
├── README.md
```

## 🛠️ Tools and Libraries
- Python
- Jupyter Notebook (VS Code)
- TensorFlow / Keras
- NumPy, OpenCV
- scikit-learn
- Matplotlib

## 📁 Dataset
- **Anime Face Dataset by Character Name**
- Contains labeled anime character face images

## 🔄 Preprocessing Steps
1. Load and resize images to 128x128
2. Normalize pixel values [0, 1]
3. Encode character labels using LabelEncoder
4. Split data into training (80%), validation (10%), and test (10%) sets
5. Save preprocessed data to `dataset.npz`

## 🧠 Model Architecture
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

## 🧪 Training Setup
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('models/best_anime_model.h5', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)
```

## ✅ Current Status
- [x] Dataset Loaded and Preprocessed
- [x] CNN Model Built
- [x] One-hot Encoded Labels
- [x] Model Compiled and Ready to Train

## 🔜 Next Steps
- Evaluate model accuracy on test set
- Run predictions on new images
- Implement behavior analysis based on character
- Build frontend UI to upload image and view results

---

Feel free to contribute or extend the behavior mapping module after classification.