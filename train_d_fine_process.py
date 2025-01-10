import copy
import os
from datetime import datetime
import yaml

from ikomia import core, dataprocess
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain

from train_d_fine.utils.ikutils import prepare_dataset
from train_d_fine.utils.download_model import get_model_path

from train_d_fine.D_FINE.src.misc import dist_utils
from train_d_fine.D_FINE.src.core import YAMLConfig
from train_d_fine.D_FINE.src.solver import TASKS


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainDFineParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        dataset_folder = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "dataset")
        self.cfg["dataset_folder"] = dataset_folder
        self.cfg["model_name"] = "dfine_m"
        self.cfg["model_weight_file"] = ""
        self.cfg["epochs"] = 80
        self.cfg["batch_size"] = 6
        self.cfg["input_size"] = 640
        self.cfg["dataset_split_ratio"] = 0.9
        self.cfg["workers"] = 0
        self.cfg["weight_decay"] = 0.000125
        self.cfg["lr"] = 0.00025
        self.cfg["config_file"] = ""
        self.cfg["output_folder"] = os.path.dirname(
            os.path.realpath(__file__)) + "/runs/"

    def set_values(self, param_map):
        self.cfg["dataset_folder"] = str(param_map["dataset_folder"])
        self.cfg["model_name"] = str(param_map["model_name"])
        self.cfg["model_weight_file"] = str(param_map["model_weight_file"])
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["input_size"] = int(param_map["input_size"])
        self.cfg["workers"] = int(param_map["workers"])
        self.cfg["weight_decay"] = float(param_map["weight_decay"])
        self.cfg["lr"] = float(param_map["lr"])
        self.cfg["config_file"] = str(param_map["config_file"])
        self.cfg["dataset_split_ratio"] = float(
            param_map["dataset_split_ratio"])
        self.cfg["output_folder"] = str(param_map["output_folder"])


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainDFine(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Add input/output of the algorithm here
        # Create parameters object
        if param is None:
            self.set_param_object(TrainDFineParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.enable_mlflow(True)
        self.cfg_folder = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "configs")
        self.experiment_name = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def set_output_dir(self, param):
        # Create output folder
        model_name = param.cfg["model_name"]
        self.experiment_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(param.cfg["output_folder"], exist_ok=True)
        output_folder = os.path.join(
            param.cfg["output_folder"], self.experiment_name)
        os.makedirs(output_folder, exist_ok=True)

        return output_folder

    def load_config(self, cfg_path):
        with open(cfg_path, 'r') as f:
            # Change to unsafe_load to handle YAML object references
            return yaml.unsafe_load(f)

    def run(self):
        param = self.get_param_object()
        dataset_input = self.get_input(0)

        # Prepare dataset
        dataset_yaml_info = prepare_dataset(dataset_input.data,
                                            param.cfg["dataset_folder"],
                                            param.cfg["dataset_split_ratio"]
                                            )
        print(f"\nFinal dataset info: {dataset_yaml_info}")

        # Initialize distributed training
        dist_utils.setup_distributed(print_rank=0, print_method='builtin')

        # Set custom configuration file
        if param.cfg["config_file"]:
            print(
                f"Using custom configuration file: {param.cfg['config_file']}")
            # cfg = YAMLConfig(param.cfg["config_file"])
            cfg = self.load_config(param.cfg["config_file"])
        else:
            # Load YAML configuration
            model_size = param.cfg["model_name"][-1]
            cfg_file = os.path.join(self.cfg_folder, "custom",
                                    f"dfine_hgnetv2_{model_size}_custom.yml")

            # Download weights and set path
            model_weights = get_model_path(param)

            # Set output folder
            output_folder = self.set_output_dir(param)

            # Load config
            cfg = YAMLConfig(
                cfg_file,
                output_dir=output_folder,
                tuning=model_weights)

            # Modify batch size, learning rate, weight decay, workers, image size
            cfg.yaml_cfg['train_dataloader']['total_batch_size'] = param.cfg["batch_size"]
            cfg.yaml_cfg['val_dataloader']['total_batch_size'] = param.cfg["batch_size"]

            cfg.yaml_cfg['optimizer']['lr'] = param.cfg["lr"]
            cfg.yaml_cfg['optimizer']['weight_decay'] = param.cfg["weight_decay"]

            cfg.yaml_cfg['train_dataloader']['num_workers'] = param.cfg["workers"]
            cfg.yaml_cfg['val_dataloader']['num_workers'] = param.cfg["workers"]

            cfg.yaml_cfg['eval_spatial_size'] = [
                param.cfg["input_size"], param.cfg["input_size"]]
            cfg.yaml_cfg['train_dataloader']['collate_fn']['base_size'] = param.cfg["input_size"]
            cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['ops'][5]['size'] = [
                param.cfg["input_size"], param.cfg["input_size"]]
            cfg.yaml_cfg['val_dataloader']['dataset']['transforms']['ops'][0]['size'] = [
                param.cfg["input_size"], param.cfg["input_size"]]

            # Set dataset paths and number of classes
            cfg.yaml_cfg['train_dataloader']['dataset']['img_folder'] = dataset_yaml_info["train_img_dir"]
            cfg.yaml_cfg['train_dataloader']['dataset']['ann_file'] = dataset_yaml_info["train_annot_file"]
            cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'] = dataset_yaml_info["val_img_dir"]
            cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = dataset_yaml_info["val_annot_file"]
            cfg.yaml_cfg['num_classes'] = dataset_yaml_info["nc"]

            # Save new configuration file (used for inference)
            training_config = os.path.join(
                output_folder, f'config_{self.experiment_name}.yaml')
            with open(training_config, 'w') as file:
                yaml.dump(cfg, file)

            # Save class names for inference
            class_names = dataset_yaml_info["names"]
            with open(os.path.join(output_folder, 'class_names.txt'), 'w') as file:
                for name in class_names:
                    file.write(f"{name}\n")

        # Disable pretrained option if HGNetv2 is in the configuration
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        # Initialize solver
        solver = TASKS[cfg.yaml_cfg['task']](cfg)

        # Train or validate
        if param.cfg.get('test_only', False):
            solver.val()
        else:
            solver.fit()

        dist_utils.cleanup()

        self.emit_step_progress()
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainDFineFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "train_d_fine"
        self.info.short_description = "Train D-FINE models"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Y. Peng, H. Li, P. Wu, Y. Zhang, X. Sun and F. Wu"
        self.info.article = "D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement"
        self.info.journal = "arXiv"
        self.info.year = 2024
        self.info.license = "Apache 2.0"

        # Python compatibility
        self.info.min_python_version = "3.11.0"

        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/pdf/2410.13842"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_d_fine"
        self.info.original_repository = "https://github.com/Peterande/D-FINE"

        # Keywords used for search
        self.info.keywords = "DETR, object, detection, real-time"
        self.info.algo_type = core.AlgoType.TRAIN
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create algorithm object
        return TrainDFine(self.info.name, param)
