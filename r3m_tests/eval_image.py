import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

from r3m import load_r3m

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
r3m.to(device)

## DEFINE PREPROCESSING
transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()]) # ToTensor() divides by 255

## ENCODE IMAGE
# image = np.random.randint(0, 255, (500, 500, 3))
image = Image.open("mug_center.jpg")
preprocessed_image = transforms(image).reshape(-1, 3, 224, 224)
# preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
preprocessed_image.to(device) 
with torch.no_grad():
  embedding_center = r3m(preprocessed_image * 255.0) ## R3M expects image input to be [0-255]

image = Image.open("mug_left.jpg")
preprocessed_image = transforms(image).reshape(-1, 3, 224, 224)
# preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
preprocessed_image.to(device) 
with torch.no_grad():
  embedding_left = r3m(preprocessed_image * 255.0) ## R3M expects image input to be [0-255]

image = Image.open("mug_right.jpg")
preprocessed_image = transforms(image).reshape(-1, 3, 224, 224)
# preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
preprocessed_image.to(device) 
with torch.no_grad():
  embedding_right = r3m(preprocessed_image * 255.0) ## R3M expects image input to be [0-255]

diff_rl = torch.linalg.norm(embedding_right - embedding_left)
diff_cl = torch.linalg.norm(embedding_center - embedding_left)
diff_cr = torch.linalg.norm(embedding_center - embedding_right)
print('diff rl', diff_rl)
print('diff cl', diff_cl)
print('diff cr', diff_cr)