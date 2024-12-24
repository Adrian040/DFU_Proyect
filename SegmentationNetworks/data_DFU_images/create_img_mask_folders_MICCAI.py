""" Crea las carpetas images y masks necesarias para la posterior división de 
los datos en train, validation y test con div_data.py ."""

import pandas as pd
import io
import os
from PIL import Image
import shutil



def count_files_in_folder(folder_path):

  if not os.path.isdir(folder_path):
    return 0  # O puedes lanzar una excepción si la ruta no es una carpeta válida
  file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

  # return print(f"La carpeta contiene {file_count} archivos.")
  return file_count



def copy_folders_content(source_folders, destination_folder):
  """
  Copies the content of multiple source folders into a new destination folder.

  Args:
    source_folders: A list of paths to the source folders.
    destination_folder: The path to the new destination folder.
  """

  if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

  for source_folder in source_folders:
    for item in os.listdir(source_folder):
      source_path = os.path.join(source_folder, item)
      destination_path = os.path.join(destination_folder, item)

      if os.path.isdir(source_path):
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
      else:
        shutil.copy2(source_path, destination_path)
# Example usage:
# source_folders = ["/path/to/folder1", "/path/to/folder2"]
# destination_folder = "/path/to/new_folder"
# copy_folders_content(source_folders, destination_folder)


# Creamos el folder con todas las imágenes del conjunto de datos, así como otro para sus respectivas máscaras:

# Juntamos las todas las imágenes.
copy_folders_content(["C:/Users/am969/Documents/HReps_DFU/wound-segmentation/data/Foot Ulcer Segmentation Challenge/train/images",             # # Carpetas fuente
                      "C:/Users/am969/Documents/HReps_DFU/wound-segmentation/data/Foot Ulcer Segmentation Challenge/validation/images"],       #
                      "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/images")  # Folder de destino
# Juntamos las todas las máscaras.
copy_folders_content(["C:/Users/am969/Documents/HReps_DFU/wound-segmentation/data/Foot Ulcer Segmentation Challenge/train/labels", 
                      "C:/Users/am969/Documents/HReps_DFU/wound-segmentation/data/Foot Ulcer Segmentation Challenge/validation/labels"], 
                      "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/masks")  # Folder de destino