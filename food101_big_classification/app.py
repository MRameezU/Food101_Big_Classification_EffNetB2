import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names by reading them from the file
with open(file="class_names.txt", mode='r') as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

### Model and transforms preparation ###
# Create model and transforms for EfficientNetB2
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=101)

# Load saved weights into the model
effnetb2.load_state_dict(
    torch.load(f="effnetb2_feature_extractor_food101_20_percent.pth",
               map_location=torch.device("cpu"))  # load model to CPU
)

### Predict function ###
def predict(img) -> Tuple[Dict, float]:
    # Start a timer
    start_time = timer()

    # Transform the input image for use with EfficientNetB2
    img = effnetb2_transforms(img).unsqueeze(0)  # unsqueeze to add the batch dimension

    # Put model into evaluation mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass transformed image through the model and turn the prediction logits into probabilities
        pred_probs = torch.softmax(input=effnetb2(img), dim=1)

    # Create a prediction label and prediction probability dictionary
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate prediction time
    end_time = timer()
    pred_time = round(end_time - start_time, 4)

    # Return prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time

### Gradio app ###
# Create title, description, and article for the Gradio interface
title = "Food101_Big_Classification"
description = "An [EfficientNetB2 feature extractor](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2) computer vision model to classify images [101 classes of food from the Food101 dataset](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/food101_class_names.txt)."
article = "Created at [https://github.com/MRameezU/Food101_Big_Classification_EffNetB2/blob/e9c502b4f9e5a0f68e102184d18a6b934334b17b/food101-big-classification-effnetb2.ipynb]"

# Create a list of example images
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo interface
demo = gr.Interface(fn=predict,  # function to map inputs to outputs
                    inputs=gr.Image(type="pil"),  # input type is an image
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"),  # output top 5 predictions
                             gr.Number(label="Prediction time (s)")],  # output prediction time
                    examples=example_list,  # provide example images
                    title=title,  # set the title of the interface
                    description=description,  # set the description
                    article=article)  # set the article link

# Launch the Gradio demo
demo.launch()
