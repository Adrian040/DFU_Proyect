# prompt: Escribe una función que tome como argumento el path a una carpeta la cual tiene imágenes que al momento de leerlas con Image.open(image_path) y convertirlas a array np.array(img) devuelve un array de True y False, necesito que la función convierta todas las imagenes de la carpeta a imagenes con formato de escala de grises (donde los True deberán ser 255 y los false 0), sobreescribiendo las imagenes
from PIL import Image
import numpy as np
import os

# (Aún no se ha probado esta función pero al parecer funciona.
# Es para conmvertir las máscaras generadas manualmente en matlab.)

def convert_images_to_grayscale(folder_path):
    """
    Converts all images in a folder to grayscale, replacing True with 255 and False with 0.

    Args:
        folder_path: The path to the folder containing the images.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add more extensions if needed
            image_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(image_path)
                img_array = np.array(img)

                # Convert True/False to 255/0
                img_array = (img_array * 255).astype(np.uint8)
                
                # Create a new image from the modified array
                new_img = Image.fromarray(img_array)

                #Overwrite the original image
                new_img.save(image_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
convert_images_to_grayscale("/content/") # Replace with your folder path