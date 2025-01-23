import os
from PIL import Image

def resize_images(input_folder, output_folder, width, height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize((width, height))
            img_resized.save(os.path.join(output_folder, filename))

# Example usage:
inp_folder = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/raw_images_68"
out_folder = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/raw_images_68_resized"
resize_images(inp_folder, out_folder, 240, 240)