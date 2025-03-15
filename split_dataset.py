import os
import random
import shutil

def split_dataset(image_dir, mask_dir, output_dir, train_ratio=0.8):
    # Create output directories
    train_images_dir = os.path.join(output_dir, 'train_images')
    train_masks_dir = os.path.join(output_dir, 'train_masks')
    eval_images_dir = os.path.join(output_dir, 'eval_images')
    eval_masks_dir = os.path.join(output_dir, 'eval_masks')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(eval_images_dir, exist_ok=True)
    os.makedirs(eval_masks_dir, exist_ok=True)
    
    # Get list of images and masks
    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))
    
    # Ensure the number of images and masks are the same
    assert len(images) == len(masks), "The number of images and masks must be the same"
    
    # Shuffle the dataset
    combined = list(zip(images, masks))
    random.shuffle(combined)
    images[:], masks[:] = zip(*combined)
    
    # Split the dataset
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    train_masks = masks[:split_index]
    eval_images = images[split_index:]
    eval_masks = masks[split_index:]
    
    # Copy files to respective directories
    for img in train_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(train_images_dir, img))
    for mask in train_masks:
        shutil.copy(os.path.join(mask_dir, mask), os.path.join(train_masks_dir, mask))
    for img in eval_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(eval_images_dir, img))
    for mask in eval_masks:
        shutil.copy(os.path.join(mask_dir, mask), os.path.join(eval_masks_dir, mask))
    
    print(f"Dataset split completed. {len(train_images)} training and {len(eval_images)} evaluation samples.")

if __name__ == "__main__":
    image_dir = r'C:\github\res-u-net\dataset\train_images'
    mask_dir = r'C:\github\res-u-net\dataset\train_masks'
    output_dir = r'C:\github\res-u-net\output'
    
    split_dataset(image_dir, mask_dir, output_dir)