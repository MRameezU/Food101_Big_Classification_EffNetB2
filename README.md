
## Food101_Big_Classification_EffNetB2

This repository contains a feature extraction classification model using EfficientNet-B2, trained on the Food101 dataset. The model leverages the powerful EfficientNet-B2 architecture to classify images into 101 different food categories with high accuracy.

### Features
- **EfficientNet-B2 Architecture**: Utilizes the EfficientNet-B2 model for efficient and accurate image classification.
- **Food101 Dataset**: Trained on the Food101 dataset, which includes 101,000 images across 101 food categories.
- **Feature Extraction**: Employs feature extraction techniques to enhance classification performance.
- **Preprocessing and Augmentation**: Includes data preprocessing and augmentation steps to improve model robustness.
- **Transfer Learning**: Fine-tuned from pre-trained EfficientNet-B2 weights to adapt to the Food101 dataset.
- **Gradio App**: The repository includes a Gradio app for easy model inference, which will be uploaded to Hugging Face Spaces as a demo.

### Project Structure
```
Food101_Big_Classification_EffNetB2.ipyb  # Contains model training and Gradio app creation
demos/
└── Food101_Big_Classification/
    ├── effnetb2_feature_extractor_food101_20_percent.pth
    ├── app.py
    ├── class_names.txt
    ├── examples/
    │   ├── example_1.jpg
    │   ├── example_2.jpg
    │   └── example_3.jpg
    ├── model.py
    └── requirements.txt
```

### Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MRameezU/Food101_Big_Classification_EffNetB2.git
   cd Food101_Big_Classification_EffNetB2
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Inference**:
   ```python
   python classify.py --image_path path_to_your_image.jpg
   ```

### Results
The model achieves impressive accuracy on the Food101 dataset, making it suitable for various food classification tasks.

### Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

### License
This project is licensed under the MIT License.

---

Feel free to tweak this further if needed! If there's anything else you'd like to add or change, just let me know.
