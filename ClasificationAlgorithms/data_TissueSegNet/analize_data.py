import os
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import exposure

def analyze_masks(mask_path):
    os.makedirs("assets_analysis", exist_ok=True)
    os.makedirs("assets_analysis/masks_analysis", exist_ok=True)
    os.makedirs("assets_analysis/images_analysis", exist_ok=True)
    
    masks = [np.array(Image.open(os.path.join(mask_path, f)).convert("P"), dtype=np.float32) for f in os.listdir(mask_path) if f.endswith('.png')]

    class_counts = {i: [] for i in range(4)}
    total_pixels = np.prod(masks[0].shape)
    
    for mask in masks:
        for i in range(4):
            class_counts[i].append(np.sum(mask == i) / total_pixels)
    
    avg_proportions = {i: np.mean(class_counts[i]) for i in range(4)}
    # avg_proportions_2 = {i: class_counts[i] for i in range(4)}
    # # Guardar avg_proportions en un archivo JSON para ver los valores individuales de cada imagen.
    # with open("assets_masks/masks_analysis/avg_class_proportions.json", "w") as f:
    #     json.dump(avg_proportions_2, f, indent=4)
    
    # Histograma de distribución de proporciones
    bins = np.linspace(0, 1, 11)
    hist_data = {i: np.histogram(class_counts[i], bins=bins)[0].tolist() for i in range(4)}
    
    # Guardar en JSON
    result = {"average_proportions": avg_proportions, "histogram": hist_data}
    with open("assets_analysis/masks_analysis/class_distribution.json", "w") as f:
        json.dump(result, f, indent=4)
    
    # Graficar histogramas
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.bar(bins[:-1], hist_data[i], width=0.09, alpha=0.7, label=f'Class {i}')
    plt.xlabel("Proportion bins")
    plt.ylabel("Image count")
    plt.title("Histogram of class proportions")
    plt.legend()
    plt.savefig("assets_analysis/masks_analysis/hist_class_distribution.png")
    plt.close()

def analyze_images(image_path):
    os.makedirs("assets_masks", exist_ok=True)
    
    images = [cv2.imread(os.path.join(image_path, f)) for f in os.listdir(image_path) if f.endswith('.png')]
    
    original_size = images[0].shape[:2]
    useful_pixel_ratios = []
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        useful_pixels = np.sum(thresh > 0)
        useful_pixel_ratios.append(useful_pixels / np.prod(original_size))
    
    avg_useful_ratio = np.mean(useful_pixel_ratios)
    
    # Histograma de proporciones de píxeles útiles
    bins = np.linspace(0, 1, 11)
    hist_data, _ = np.histogram(useful_pixel_ratios, bins=bins)
    
    result = {"average_useful_pixel_ratio": avg_useful_ratio, "histogram": hist_data.tolist()}
    with open("assets_analysis/images_analysis/image_size_analysis.json", "w") as f:
        json.dump(result, f, indent=4)
    
    # Graficar histograma
    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1], hist_data, width=0.09, alpha=0.7)
    plt.xlabel("Useful pixel proportion bins")
    plt.ylabel("Image count")
    plt.title("Histogram of useful pixel proportions")
    plt.savefig("assets_analysis/images_analysis/hist_pixel_proportion.png")
    plt.close()
    
    # Métricas de color, brillo, textura y contexto
    color_metrics = []
    brightness_metrics = []
    texture_metrics = []
    
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        brightness_metrics.append(brightness)
        
        r, g, b = cv2.split(img)
        color_metrics.append({
            "red": np.mean(r), "green": np.mean(g), "blue": np.mean(b),
            "std_red": np.std(r), "std_green": np.std(g), "std_blue": np.std(b)
        })
        
        gray = rgb2gray(img)
        glcm = graycomatrix((gray * 255).astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        texture_metrics.append({"contrast": contrast, "dissimilarity": dissimilarity})
    
    result_metrics = {
        "average_brightness": np.mean(brightness_metrics),
        "average_color": {
            "red": np.mean([c["red"] for c in color_metrics]),
            "green": np.mean([c["green"] for c in color_metrics]),
            "blue": np.mean([c["blue"] for c in color_metrics])
        },
        "average_texture": {
            "contrast": np.mean([t["contrast"] for t in texture_metrics]),
            "dissimilarity": np.mean([t["dissimilarity"] for t in texture_metrics])
        }
    }
    
    with open("assets_analysis/images_analysis/image_metrics.json", "w") as f:
        json.dump(result_metrics, f, indent=4)

# Ejemplo de uso
# analyze_masks("path/to/masks")
# analyze_images("path/to/images")
if __name__ == "__main__":
    analyze_masks("data_padded/masks")
    analyze_images("data_padded/images")