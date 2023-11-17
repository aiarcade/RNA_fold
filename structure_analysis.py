from rnadatasets import *
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch

dataset=StructureProbDataset("../2A3_MaP")
images=[]
i=0

for img,r in dataset:
    to_pil=transforms.ToPILImage()
    pil_image = to_pil(img*255)
    images.append(pil_image)
    sample=[img,r]
    #pil_image.save("output_image2a3_"+str(i)+".png")
    if i==4:
        break
    else:
        i=i+1


dataset=StructureProbDataset("../DMS_MaP")
i=0
for img,r in dataset:
    to_pil=transforms.ToPILImage()
    pil_image = to_pil(img*255)
    images.append(pil_image)
    #pil_image.save("output_imagedms_"+str(i)+".png")
    if i==4:
        break
    else:
        i=i+1
grid_width = 5  # Number of images per row in the grid
grid_height = (len(images) + grid_width - 1) // grid_width
grid_size = (grid_width * images[0].width, grid_height * images[0].height)
grid_image = Image.new('RGB', grid_size, (255, 255, 255))

# Paste each image into the grid
for i, image in enumerate(images):
    row = i // grid_width
    col = i % grid_width
    grid_image.paste(image, (col * image.width, row * image.height))

# Save the grid image
grid_image.save("grid_image.png")

combines=sample[0]*sample[1]

numpy_image = combines.numpy()

# Create a heatmap using matplotlib
plt.imshow(numpy_image, cmap='hot', interpolation='nearest')
plt.axis('off')  # Turn off axis labels
plt.colorbar()   # Add a colorbar

# Save the heatmap as an image
plt.savefig('heatmap.png', bbox_inches='tight', pad_inches=0.0)
plt.close()  # Close the plot to prevent it from being displayed


count_above_one = torch.sum(sample >= .5).item()

print(f"Number of values greater than or equal to 1: {count_above_one}")