import os
from PIL import Image

def resize_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                resized_img = img.resize((240, 240))
                resized_img.save(image_path)

# Example usage:
resize_images_in_folder('C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_semisup_padded/Resized/unlabel_data_padded')