import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple,Dict

# Setup class names
with open(file="class_names.txt",mode='r') as f:
  class_names=[food_name.strip() for food_name in f.readlines()]

### 2. Model and transforms preparation ### 
# Create model and transforms

effnetb2,effnetb2_transforms = create_effnetb2_model(num_classes=101)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(f="09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth",
    map_location=torch.device("cpu")) # load model to CPU
)

### 3. preduct function
def predict(img)->Tuple[Dict,float]:
  # start a timer
  start_time=timer()

  # Transform the input image for use with EffNetB2
  img=effnetb2_transforms(img).unsqueeze(0) # nsqueeze =add the batch dimension

  # put model into eval mode
  effnetb2.eval()
  with torch.inference_mode():
    # Pass transformed image through the model and turn the prediction logits into probaiblities
    pred_probs=torch.softmax(input=effnetb2(img),dim=1)

  # Create a prediction label and prediction probability dictionary
  pred_labels_and_probs = {class_names[i]:float(pred_probs[0][i] for i in range(len(class_names)))}

  # Calculate pred time
  end_time=timer()
  pred_time=round(end_time-start_time,4)

  # return pred dict and pred time
  return pred_labels_and_probs,pred_time

### 4. Gradio app ###
# Create title, description and article
title ="FoodVision BIG"
description="An [EfficientNetB2 feature extractor](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2) computer vision model to classify images [101 classes of food from the Food101 dataset](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/food101_class_names.txt)."
article="Created at [https://github.com/MRameezU/Food101_Big_Classification_EffNetB2/blob/e9c502b4f9e5a0f68e102184d18a6b934334b17b/food101-big-classification-effnetb2.ipynb]"

# create example list
example_list=[["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # maps inputs to outputs
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch() 
