from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'

image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

input = processor(image, return_tensors="pt")

with torch.no_grad():
    out = model.generate(**input)
    
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)