import os
from torchvision import transforms
from PIL import Image


from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("../openai/clip-vit-large-patch14/")
processor = CLIPProcessor.from_pretrained("../openai/clip-vit-large-patch14/")

images = []
path = "../code-slicer/test/"
for filename in os.listdir(path):
    new_path = os.path.join(path,filename)
    image = Image.open(new_path)
    images.append(image)

transform = transforms.ToTensor()
images = [transform(img) for img in images]
inputs = processor(text=["a photo of a non-vulnerability code", "a photo of a vulnerability code"], images=images, return_tensors="pt", padding=True)

outputs = model(**inputs)
print(outputs.vision_model_output.last_hidden_state.shape)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print(probs)