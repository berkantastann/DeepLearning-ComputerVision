import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path,max_size=400,shape=None):
    image = Image.open(image_path).convert('RGB')
    
    if shape is not None:
        size = shape
    else:
        size = max(image.size)
        if size > max_size:
            size = max_size
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
    
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image.to(device) 

def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor((0.229, 0.224, 0.225)).view(3,1,1)
    image = image + torch.tensor((0.485, 0.456, 0.406)).view(3,1,1)
    image = image.clamp(0, 1)
    
    return image.permute(1, 2, 0).numpy()

def gramm_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    
    return gram

class VGGFeatures(nn.Module):
    def __init__(self, weight_path="./models/vgg19-dcbb9e9d.pth"):
        super(VGGFeatures, self).__init__()
        
        # VGG19'i manuel yükleme
        vgg_model = models.vgg19()
        if os.path.exists(weight_path):
            vgg_model.load_state_dict(torch.load(weight_path, map_location=device))
        else:
            raise FileNotFoundError(f"Model weight not found at {weight_path}")
        
        self.vcc = nn.Sequential(*list(vgg_model.features.children())[:29]).to(device).eval()
        for param in self.vcc.parameters():
            param.requires_grad = False
        
        self.style_layers = {'0': 'conv1_1',
                             '5': 'conv2_1',
                             '10': 'conv3_1',
                             '19': 'conv4_1',
                             '21': 'conv4_2',
                             '28': 'conv5_1'}
        
    def forward(self, x):
        features = {}
        for name, layer in self.vcc._modules.items():
            x = layer(x)
            if name in self.style_layers:
                features[self.style_layers[name]] = x
        return features

content = load_image("content.jpg", max_size=400)
style = load_image("style.jpg", shape=[content.size(2), content.size(3)])

target = content.clone().requires_grad_(True).to(device)

vgg = VGGFeatures(weight_path="vgg19-dcbb9e9d.pth")

style_weights = {'conv1_1': 1.0,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}
content_weight = 1e4
style_weight = 1e2

style_features = vgg(style)
content_features = vgg(content)

style_grams = {layer: gramm_matrix(style_features[layer]) for layer in style_features}

optimizer = optim.Adam([target], lr=0.003)
steps = 1000  # iterasyon sayısı

for i in tqdm.tqdm(range(steps)):
    target_features = vgg(target)
    
   
    content_loss = content_weight * torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
  
    style_loss = 0
    for layer in style_weights:
        target_gram = gramm_matrix(target_features[layer])
        style_gram = style_grams[layer]
        layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_loss * style_weight
    
    # Toplam kayıp
    total_loss = content_loss + style_loss
    
    # Backprop ve step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Her 50 iterasyonda kaybı yazdır
    if i % 50 == 0:
        print(f"Step [{i}/{steps}], Total Loss: {total_loss.item():.2f}")

final_img = im_convert(target)
plt.imshow(final_img)
plt.axis('off')
plt.show()