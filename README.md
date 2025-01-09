<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">train_d_fine</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_d_fine">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_d_fine">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_d_fine/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_d_fine.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Train D-FINE object detection models.

Following best practices recommended by the [official repository](https://github.com/Peterande/D-FINE), this algorithm utilizes the Objects365 pre-trained model as a foundation for fine-tuning, enabling optimized performance for custom object detection tasks.

![Desk object detection](https://raw.githubusercontent.com/Ikomia-hub/train_d_fine/main/images/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow
```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "detection",
}) 

# Add training algorithm
train = wf.add_task(name="train_d_fine", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).


## :pencil: Set algorithm parameters
- `model_name` (str) - default 'dfine_m': Name of the D-FINE pre-trained model. Other model available:
    - dfine_n
    - dfine_s
    - dfine_l
    - dfine_x
- `batch_size` (int) - default '8': Number of samples processed before the model is updated.
- `epochs` (int) - default '100': Number of complete passes through the training dataset.
- `dataset_split_ratio` (float) – default '0.9': Divide the dataset into train and evaluation sets ]0, 1[.
- `input_size` (int) - default '640': Size of the input image.
- `weight_decay` (float) - default '0.000125': Amount of weight decay, regularization method.
- `workers` (int) - default '0': Number of worker threads for data loading (per RANK if DDP).
- `lr` (float) - default '0.00025': Initial learning rate . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
- `output_folder` (str, *optional*): path to where the model will be saved. 
- `config_file` (str, *optional*): path to the training config file .yaml. Using a [config file](https://github.com/Ikomia-hub/train_d_fine/configs/template/config_template.yaml) allows you to set all the train settings available. 

**Parameters** should be in **strings format**  when added to the dictionary.
```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "detection",
}) 

# Add training algorithm
train = wf.add_task(name="train_d_fine", auto_connect=True)

train.set_parameters({
    "model_name": "dfine_m",
    "epochs": "100",
    "batch_size": "6",
    "input_size": "640",
    "dataset_split_ratio": "0.9",
    "workers": "0",  # Recommended to set to 0 if you are using Windows
    "weight_decay": "0.000125",
    "lr": "0.00025",
    "output_folder": "Path/To/Output/Folder", # Default folder : runs 
    "model_weight_file": "", # Optional
    "config_file": "Path/To/Config/file", # Optional 
    
})

# Launch your training on your data
wf.run()
```
