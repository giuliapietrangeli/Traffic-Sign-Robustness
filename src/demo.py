import gradio as gr
import torch
import albumentations as A
import numpy as np
from PIL import Image
from model import get_model
from data_loader import get_transforms

# GTSRB Class Names
CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Load Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'

model = get_model(num_classes=43)
try:
    # Load the robust model by default
    model.load_state_dict(torch.load('models/model_robust.pth', map_location=device))
    print("Loaded Robust Model")
except:
    print("Could not load robust model, check path.")

model.to(device)
model.eval()

def predict(image, weather):
    if image is None:
        return None, None
    
    # Convert to PIL if needed (Gradio passes numpy array usually)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # Apply Augmentation
    transform = None
    if weather == 'Rain':
        transform = get_transforms(specific_corruption='rain')
    elif weather == 'Fog':
        transform = get_transforms(specific_corruption='fog')
    elif weather == 'Blur':
        transform = get_transforms(specific_corruption='blur')
    else: # Clean
        transform = get_transforms(augment=False)
        
    # Transform returns tensor, but we also want to show the augmented image
    # So we need to do a bit of a trick: apply albumentations part first to get image, then normalize
    
    # Let's manually apply the specific corruption using albumentations logic from data_loader
    # We'll recreate the transform pipeline just for the image part
    
    aug_transform = []
    aug_transform.append(A.Resize(32, 32))
    
    if weather == 'Rain':
        aug_transform.append(A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0))
    elif weather == 'Fog':
        aug_transform.append(A.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.1, p=1.0))
    elif weather == 'Blur':
        aug_transform.append(A.MotionBlur(blur_limit=5, p=1.0))
        
    aug_pipeline = A.Compose(aug_transform)
    
    # Apply to image
    img_np = np.array(image)
    augmented = aug_pipeline(image=img_np)['image']
    
    # Prepare for model (Normalize + ToTensor)
    norm_pipeline = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2()
    ])
    
    tensor = norm_pipeline(image=augmented)['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Get top 3
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    
    confidences = {CLASSES[idx.item()]: float(prob) for idx, prob in zip(top3_idx, top3_prob)}
    
    return augmented, confidences

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload Traffic Sign"),
        gr.Radio(["Clean", "Rain", "Fog", "Blur"], label="Weather Condition", value="Clean")
    ],
    outputs=[
        gr.Image(label="Processed Image (32x32)"),
        gr.Label(num_top_classes=3, label="Prediction")
    ],
    title="Robust Traffic Sign Classifier",
    description="Upload a traffic sign image. Apply synthetic weather effects and see how the Robust ResNet18 model classifies it.",
    examples=[
        # We can't easily put local paths here without knowing what's available, 
        # but users can upload their own.
    ]
)

if __name__ == "__main__":
    iface.launch()
