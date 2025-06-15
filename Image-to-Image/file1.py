# cgan_pix2pix.py

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from models import Generator  # Pre-trained generator class

# pip install torch torchvision

# Load pre-trained model checkpoint (Assume already trained)
model_path = "models/pix2pix_generator.pth"  # <-- Put your model file here

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

input_image = Image.open("data/input.jpg").convert("RGB")
input_tensor = transform(input_image).unsqueeze(0).to(device)

# Generate output
with torch.no_grad():
    fake_image = generator(input_tensor)

# Save result
save_image((fake_image + 1) / 2, "output/generated_image.png")
print("✅ Image-to-Image translation complete!")
