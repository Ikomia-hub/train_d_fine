from train_d_fine.D_FINE.src.core import YAMLConfig
import os
import urllib.request
import torch


def get_config_path(root_dir, model_name, dataset):
    # Extract model size from the last character of model_name
    model_size = model_name[-1]

    # Define the base configuration directory
    root_config = os.path.join(root_dir, "D_FINE", "configs", "dfine")

    # Handle the case where model_size is 'n' and it's not a coco dataset
    if model_size == 'n' and dataset != "coco":
        raise ValueError("Model size 'n' is only valid for the coco dataset.")

    # Determine the appropriate configuration file based on the dataset
    if dataset == "obj2coco":
        config_file = os.path.join(
            root_config, "objects365", f"dfine_hgnetv2_{model_size}_obj2coco.yml")
    elif dataset == "coco":
        config_file = os.path.join(
            root_config, f"dfine_hgnetv2_{model_size}_coco.yml")
    elif dataset == "obj365":
        config_file = os.path.join(
            root_config, "objects365", f"dfine_hgnetv2_{model_size}_obj365.yml")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return config_file


def get_model_path(param):
    """
    Determine the path to the model weights file, downloading it if necessary.

    Args:
        param: An object with the following attributes:
            - model_weight_file (str): Path to the specific model weight file.
            - model_name (str): Name of the model.

    Returns:
        str: Path to the model weights file, or None if an error occurs.
    """
    # If a specific model weight file is provided
    if param.cfg["model_weight_file"]:
        if os.path.isfile(param.cfg["model_weight_file"]):
            return param.model_weight_file
        else:
            print("Invalid model weights path provided. Ensure the file exists.")
            return None

    # Define default model path
    current_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(current_dir)
    model_folder = os.path.join(root_dir, "weights")
    model_name = param.cfg["model_name"]
    model_weights = os.path.join(model_folder, f"{model_name}_obj365.pth")

    # Ensure the weights directory exists
    os.makedirs(model_folder, exist_ok=True)

    # If the model file doesn't exist, download it
    if not os.path.isfile(model_weights):
        url = f"https://github.com/Peterande/storage/releases/download/dfinev1.0/{model_name}_obj365.pth"
        print(f"Downloading model weights from {url}...")

        try:
            urllib.request.urlretrieve(url, model_weights)
            print(f"Model weights downloaded and saved to {model_weights}")
        except Exception as e:
            print(f"Failed to download the model weights: {e}")
            return None

    return model_weights
