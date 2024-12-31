import sys
import os
# Añadir la ruta del directorio raíz del proyecto al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'C:/Users/am969/Documents/DFU_Proyect')); sys.path.append(project_root)
from SegmentationNetworks.data_DFU_images.create_img_mask_folders_MICCAI import count_files_in_folder, copy_folders_content

x = count_files_in_folder("C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/images")
print(x)

# Juntamos las todas las imágenes.
copy_folders_content(["C:/Users/am969/Documents/HReps_DFU/DFUTissueSegNet/DFUTissue/Labeled/Padded/Images/Test",             # # Carpetas fuente
                      "C:/Users/am969/Documents/HReps_DFU/DFUTissueSegNet/DFUTissue/Labeled/Padded/Images/TrainVal"],       #
                      "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/images")  # Folder de destino
# Juntamos las todas las máscaras.
copy_folders_content(["C:/Users/am969/Documents/HReps_DFU/DFUTissueSegNet/DFUTissue/Labeled/Padded/Palette/Test", 
                      "C:/Users/am969/Documents/HReps_DFU/DFUTissueSegNet/DFUTissue/Labeled/Padded/Palette/TrainVal"], 
                       "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/masks")  # Folder de destino

print(count_files_in_folder("C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/images"), count_files_in_folder("C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/masks"))