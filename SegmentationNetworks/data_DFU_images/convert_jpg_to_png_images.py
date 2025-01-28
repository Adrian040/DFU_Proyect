from PIL import Image
import os

def convert_jpg_to_png(folder_path):
  """
  Converts all .jpg images in a folder to .png format.

  Args:
    folder_path: The path to the folder containing the .jpg images.
  """
  if not os.path.exists(folder_path):
    print(f"Error: Folder '{folder_path}' not found.")
    return

  for filename in os.listdir(folder_path):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
      img_path = os.path.join(folder_path, filename)
      try:
        img = Image.open(img_path)
        png_filename = os.path.splitext(filename)[0] + ".png"
        png_path = os.path.join(folder_path, png_filename)
        img.save(png_path)
        print(f"Converted '{filename}' to '{png_filename}'")
        os.remove(img_path)
      except IOError as e:
        print(f"Error converting '{filename}': {e}")

# Example usage:  Replace with your folder path
convert_jpg_to_png("C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/raw_images_68")