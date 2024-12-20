import pandas as pd
import os
from PIL import Image

def convert_to_yolo_format(df, images_dir, output_dir):
    # Create the directory for annotations if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, row['file_name'])
        image = Image.open(image_path)
        width, height = image.size

        # Get the bounding box in YOLO format (normalized)
        xmin, ymin, xmax, ymax = row['x'], row['y'], row['x'] + row['width'], row['y'] + row['height']
        
        # Normalize the bounding box
        center_x = (xmin + xmax) / 2 / width
        center_y = (ymin + ymax) / 2 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        # Write to the corresponding .txt file
        label_path = os.path.join(output_dir, row['file_name'].replace('.jpg', '.txt'))  # Assuming images are JPG
        with open(label_path, 'w') as f:
            f.write(f"0 {center_x} {center_y} {bbox_width} {bbox_height}\n")  # Class '0' for vehicle

if __name__ == "__main__":
    # Paths
    csv_path = 'C:/Users/mucha/Desktop/vehicle_detection/train_dataset.csv'
    images_dir = 'C:/Users/mucha/Desktop/vehicle_detection/DataSet/train'
    output_dir = 'C:/Users/mucha/Desktop/vehicle_detection/annotations'
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Convert to YOLO format
    convert_to_yolo_format(df, images_dir, output_dir)
    print(f"Annotations saved in {output_dir}")
