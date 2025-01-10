import os
import json
import shutil
import random
import yaml  # pip install pyyaml


def create_coco_format_dict(images_list, categories_map):
    """
    Given a list of image dicts (each with 'annotations') and a categories_map
    from the Deep Learning Dataset Ikomia format, this function will
    convert them into a standard COCO-format dictionary.
    """
    coco_images = []
    coco_annotations = []

    # Convert category map -> COCO categories list
    coco_categories = []
    for cat_id, cat_name in categories_map.items():
        coco_categories.append({
            "id": cat_id,
            "name": cat_name
        })

    annotation_id = 0
    for img_info in images_list:
        # Build COCO "images" entry
        coco_images.append({
            "id": img_info["image_id"],
            "file_name": os.path.basename(img_info["filename"]),
            "width": img_info["width"],
            "height": img_info["height"]
        })

        # Build COCO "annotations" entries
        for ann in img_info["annotations"]:
            x, y, w, h = ann["bbox"]  # assume these are already [x, y, w, h]
            area = w * h
            coco_annotations.append({
                "id": annotation_id,
                "image_id": img_info["image_id"],
                "category_id": ann["category_id"],
                "bbox": [x, y, w, h],
                "iscrowd": ann["iscrowd"],
                "area": area,
                "segmentation": ann.get("segmentation_poly", [])
            })
            annotation_id += 1

    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories
    }


def _load_dataset_yaml(yaml_path):
    """
    Reads dataset.yaml and returns a dictionary with:
      {
        "train_annot_file": str,
        "train_img_dir": str,
        "val_annot_file": str,
        "val_img_dir": str,
        "nc": int,
        "names": list of str
      }
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    # Return a dict with the 6 keys we need.
    return {
        "train_annot_file": data["train_annotation_file"],
        "train_img_dir": data["train_image_dir"],
        "val_annot_file": data["val_annotation_file"],
        "val_img_dir": data["val_image_dir"],
        "nc": data["nc"],
        "names": data["names"]
    }


def _create_dataset_yaml(
    dataset_folder,
    train_annot_file,
    train_img_dir,
    val_annot_file,
    val_img_dir,
    categories_map
):
    """
    Creates dataset.yaml containing the 4 paths plus nc and names:
      train_annotation_file, train_image_dir,
      val_annotation_file,   val_image_dir,
      nc,                    names
    """
    yaml_data = {
        "train_annotation_file": train_annot_file,
        "train_image_dir": train_img_dir,
        "val_annotation_file": val_annot_file,
        "val_image_dir": val_img_dir,
        "nc": len(categories_map),
        "names": list(categories_map.values())
    }

    yaml_path = os.path.join(dataset_folder, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    return yaml_path


def _dataset_exists(data_dict, dataset_folder, split_ratio):
    """
    check to see if a COCO-style dataset folder already exists
    and matches the data in 'data_dict' (including class count, image count, etc.).
    If mismatch is found, the entire 'dataset_folder' will be removed to rebuild.
    """

    dataset_yaml_path = os.path.join(dataset_folder, "dataset.yaml")
    print("Checking dataset.yaml at:", dataset_yaml_path)

    if not os.path.isfile(dataset_yaml_path):
        print("dataset.yaml not found at that absolute path.")
        return False

    try:
        # Load the existing dataset.yaml
        existing_info = _load_dataset_yaml(dataset_yaml_path)

        # Check if the annotation files exist
        train_annot_file = existing_info["train_annot_file"]
        val_annot_file = existing_info["val_annot_file"]
        if not os.path.exists(train_annot_file) or not os.path.exists(val_annot_file):
            shutil.rmtree(dataset_folder)
            return False

        # Check if the image folders exist
        train_img_dir = existing_info["train_img_dir"]
        val_img_dir = existing_info["val_img_dir"]
        if not os.path.exists(train_img_dir) or not os.path.exists(val_img_dir):
            shutil.rmtree(dataset_folder)
            return False

        # Check if the number of classes (nc) matches
        categories_map = data_dict["metadata"]["category_names"]
        if len(categories_map) != existing_info["nc"]:
            shutil.rmtree(dataset_folder)
            return False

        # Check if len(names) matches
        if len(categories_map) != len(existing_info["names"]):
            shutil.rmtree(dataset_folder)
            return False

        # Check the number of images in train/val folders
        images = data_dict["images"]
        val_size = int((1 - split_ratio) * len(images))
        train_size = int(len(images) - val_size)

        train_files_count = int(len(os.listdir(train_img_dir)))
        val_files_count = int(len(os.listdir(val_img_dir)))

        if train_files_count != train_size or val_files_count != val_size:
            shutil.rmtree(dataset_folder)
            return False

        # If everything checks out, return True
        return True

    except Exception as e:
        # If any error arises, we remove and return False
        print(f"Error reading or validating existing dataset: {e}")
        shutil.rmtree(dataset_folder)
        return False


def prepare_dataset(data_dict, dataset_folder, split_ratio):
    """
    Main function that:
     1) Checks if a valid dataset exists (_dataset_exists).
     2) If valid, prints the message, loads dataset.yaml, returns the 6-key dict.
     3) Otherwise, splits data_dict => train/valid, creates COCO annotation files,
        copies images, writes dataset.yaml, returns the 6-key dict.

    The returned dict has:
        {
          "train_annot_file": str,
          "train_img_dir": str,
          "val_annot_file": str,
          "val_img_dir": str,
          "nc": int,
          "names": list of str
        }
    """

    # 1) Check if a valid dataset already exists
    if _dataset_exists(data_dict, dataset_folder, split_ratio):
        # Show message, load and return existing dataset.yaml info
        print("A valid dataset structure already exists, skip building a new one.")
        existing_dataset_info = _load_dataset_yaml(
            os.path.join(dataset_folder, "dataset.yaml")
        )
        return existing_dataset_info

    # 2) Otherwise, build from scratch
    print("Preparing COCO-style dataset...")

    # Make sure dataset folder exists (re-created if just removed)
    os.makedirs(dataset_folder, exist_ok=True)

    # Subfolders
    train_img_dir = os.path.join(dataset_folder, "train", "images")
    val_img_dir = os.path.join(dataset_folder, "valid", "images")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    # Shuffle images before splitting if desired
    images_list = data_dict["images"]
    random.seed(42)
    random.shuffle(images_list)

    split_index = int(len(images_list) * split_ratio)
    train_images_list = images_list[:split_index]
    val_images_list = images_list[split_index:]

    # Convert to COCO
    categories_map = data_dict["metadata"]["category_names"]
    train_coco_dict = create_coco_format_dict(
        train_images_list, categories_map)
    val_coco_dict = create_coco_format_dict(val_images_list, categories_map)

    # File paths for COCO JSON
    train_annot_file = os.path.join(
        dataset_folder, "train", "train_coco_annotation.json")
    val_annot_file = os.path.join(
        dataset_folder, "valid", "valid_coco_annotation.json")

    # Write COCO annotations
    with open(train_annot_file, "w", encoding="utf-8") as f:
        json.dump(train_coco_dict, f, indent=2)
    with open(val_annot_file, "w", encoding="utf-8") as f:
        json.dump(val_coco_dict, f, indent=2)

    # Copy images
    for img_info in train_images_list:
        src = img_info["filename"]
        dst = os.path.join(train_img_dir, os.path.basename(src))
        shutil.copy2(src, dst)
    for img_info in val_images_list:
        src = img_info["filename"]
        dst = os.path.join(val_img_dir, os.path.basename(src))
        shutil.copy2(src, dst)

    print("Dataset successfully split and converted to COCO format.")
    print(
        f"Train images: {len(train_images_list)}, Valid images: {len(val_images_list)}")
    print(f"Train annotations saved to: {train_annot_file}")
    print(f"Valid annotations saved to: {val_annot_file}")

    # Create dataset.yaml
    dataset_yaml_path = _create_dataset_yaml(
        dataset_folder,
        train_annot_file,
        train_img_dir,
        val_annot_file,
        val_img_dir,
        categories_map
    )

    # Build the 6-key dict to return
    result_dict = {
        "train_annot_file": train_annot_file,
        "train_img_dir": train_img_dir,
        "val_annot_file": val_annot_file,
        "val_img_dir": val_img_dir,
        "nc": len(categories_map),
        "names": list(categories_map.values())
    }

    return result_dict
