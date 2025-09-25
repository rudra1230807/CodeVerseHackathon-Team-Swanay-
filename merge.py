import os
import shutil
from glob import glob
import random

# ==========================
# Dataset paths and class IDs
# ==========================
datasets = {
    "mussels": ("/Users/rudra/Downloads/Mussels", 0),
    "barnacles": ("/Users/rudra/Downloads/barnacle", 1),
    "corrosion": ("/Users/rudra/Downloads/Corrosion", 2),
    "algae": ("/Users/rudra/Downloads/algae", 3)
}

# ==========================
# Main dataset directories
# ==========================
base_dir = "/Users/rudra/SY/CodeVerse/p2/SmartHull_Inspector/datasets"
train_img_dir = os.path.join(base_dir, "train/images")
train_label_dir = os.path.join(base_dir, "train/labels")
val_img_dir   = os.path.join(base_dir, "val/images")
val_label_dir = os.path.join(base_dir, "val/labels")
test_img_dir  = os.path.join(base_dir, "test/images")
test_label_dir= os.path.join(base_dir, "test/labels")

# Make sure directories exist
for folder in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
    os.makedirs(folder, exist_ok=True)

# ==========================
# Split ratios
# ==========================
train_ratio = 0.7
val_ratio   = 0.2
test_ratio  = 0.1

# ==========================
# Function to copy dataset
# ==========================
def copy_dataset(dataset_path, class_id, class_name):
    # Recursively get all images
    images = glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)
    images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    n = len(images)
    if n == 0:
        print(f"⚠️ No images found in dataset '{class_name}' at {dataset_path}")
        return
    
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val  # ensure all images are used
    
    train_images = images[:n_train]
    val_images   = images[n_train:n_train+n_val]
    test_images  = images[n_train+n_val:]
    
    def copy_files(file_list, img_dest, label_dest):
        for img_path in file_list:
            filename = os.path.basename(img_path)
            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(os.path.dirname(img_path).replace("images", "labels"), label_file)
            
            # Copy image
            shutil.copy(img_path, os.path.join(img_dest, filename))
            
            # Copy label if exists, modify class ID; else create empty label
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = str(class_id)
                        new_lines.append(" ".join(parts))
                with open(os.path.join(label_dest, label_file), 'w') as f:
                    f.write("\n".join(new_lines))
            else:
                open(os.path.join(label_dest, label_file), 'a').close()
    
    # Copy to folders
    copy_files(train_images, train_img_dir, train_label_dir)
    copy_files(val_images, val_img_dir, val_label_dir)
    copy_files(test_images, test_img_dir, test_label_dir)
    
    print(f"✅ Dataset '{class_name}' merged. Total images: {n}")

# ==========================
# Merge all datasets
# ==========================
for class_name, (path, class_id) in datasets.items():
    copy_dataset(path, class_id, class_name)

print("\n🎉 All datasets merged successfully!")
