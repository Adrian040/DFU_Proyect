from PIL import Image
import numpy as np

def analyze_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Get image format
        img_format = img.format
        print("Modo: ",img.mode)
        
        # Convert image to numpy array
        img_array = np.array(img)
        print(img_array.shape)
        # Get min, max and unique values
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        unique_vals, counts = np.unique(img_array, return_counts=True)
        num_classes = len(unique_vals)
        
        # Create a dictionary to map unique values to their counts
        value_counts = dict(zip(unique_vals, counts))
        
        return {
            'format': img_format,
            'min_value': min_val,
            'max_value': max_val,
            'num_classes': num_classes,
            'unique_values': unique_vals,
            'value_counts': value_counts
        }

# Example usage
result = analyze_image("C:/Users/am969/Documents/HReps_DFU/DFUTissueSegNet/DFUTissue/Labeled/Padded/Palette/TrainVal/0940.png")
print(result)