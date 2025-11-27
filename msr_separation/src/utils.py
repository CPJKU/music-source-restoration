import os
import sys
import datetime
import logging
from typing import Dict, List, NoReturn
import yaml
import importlib
import pytz
import torch

# LABELS for FiLM conditioning
LABELS = ["vocals", "bass", "drums", "other"]

def ignore_warnings():
    import warnings

    # Filter out warnings containing "deprecated"
    warnings.filterwarnings("ignore", message=".*invalid value encountered in cast.*")
    warnings.filterwarnings("ignore", message=".*deprecated.*")
    warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
    warnings.filterwarnings("ignore", message=".*that has Tensor Cores.*")
    warnings.filterwarnings("ignore", message=".*cudnnFinalize Descriptor Failed.*")
    warnings.filterwarnings("ignore", message=".*torch.use_deterministic_algorithms(True, warn_only=True)*")
    from transformers import logging
    logging.set_verbosity_error()


logging_levels = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET
}


def logging_setup(directory, file_log_level="INFO", console_log_level='DEBUG', timezone='Europe/Vienna'):
    file_log_level = logging_levels[file_log_level]
    console_log_level = logging_levels[console_log_level]
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # create directory, generate logfilename
    os.makedirs(directory, exist_ok=True)
    a = datetime.datetime.now().astimezone(pytz.timezone(timezone))
    logfilename = '%04d%02d%02d_%02dh%02d.log' % (a.year, a.month, a.day, a.hour, a.minute)
    log_full_path = os.path.join(directory, logfilename)

    # setup log to file
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler = logging.FileHandler(log_full_path)
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # print to console
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s (%(processName)s): %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)  # always debug level
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)


def parse_yaml(config_yaml) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str or None): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


def initialize_config(module_cfg, reload=False):
    if reload and module_cfg["module"] in sys.modules:
        module = importlib.reload(sys.modules[module_cfg["module"]])
    else:
        module = importlib.import_module(module_cfg["module"])
    if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
    return getattr(module, module_cfg["main"])()


def inject_dataset_paths_into_config(config, dataset_path_4_sources=None, dataset_path_8_sources=None, dataset_path_msr_bench=None):
    """
    Inject dataset paths into dataset configs.
    
    Args:
        config: Configuration dict (can be modified in place)
        dataset_path_4_sources: Path for 4-source dataset (required if not in config)
        dataset_path_8_sources: Path for 8-source dataset (required if not in config)
        dataset_path_msr_bench: Path for MSRBench dataset (required if not in config)
    """
    # Get paths from provided values or config (required, no defaults)
    if dataset_path_4_sources is None:
        if 'dataset_path_4_sources' not in config:
            raise ValueError("dataset_path_4_sources must be provided or specified in config")
        dataset_path_4_sources = config['dataset_path_4_sources']
    
    if dataset_path_8_sources is None:
        if 'dataset_path_8_sources' not in config:
            raise ValueError("dataset_path_8_sources must be provided or specified in config")
        dataset_path_8_sources = config['dataset_path_8_sources']
    
    if dataset_path_msr_bench is None:
        if 'dataset_path_msr_bench' not in config:
            raise ValueError("dataset_path_msr_bench must be provided or specified in config")
        dataset_path_msr_bench = config['dataset_path_msr_bench']
    
    def inject_path(dataset_config):
        """Inject the appropriate dataset path into a dataset config."""
        if 'args' not in dataset_config:
            return
        
        module_name = dataset_config.get('module', '')
        if 'dataset_4s' in module_name:
            dataset_config['args']['root'] = dataset_path_4_sources
        elif 'dataset_8s' in module_name:
            dataset_config['args']['root'] = dataset_path_8_sources
        elif 'msr_bench' in module_name:
            dataset_config['args']['root'] = dataset_path_msr_bench
    
    # Inject paths into all dataset configs
    if 'datamodule' in config and 'args' in config['datamodule']:
        if 'train_dataloader' in config['datamodule']['args'] and 'dataset' in config['datamodule']['args']['train_dataloader']:
            inject_path(config['datamodule']['args']['train_dataloader']['dataset'])
        if 'val_dataloader' in config['datamodule']['args'] and 'dataset' in config['datamodule']['args']['val_dataloader']:
            inject_path(config['datamodule']['args']['val_dataloader']['dataset'])
        if 'test_dataloader' in config['datamodule']['args'] and 'dataset' in config['datamodule']['args']['test_dataloader']:
            inject_path(config['datamodule']['args']['test_dataloader']['dataset'])


def lightning_load_from_checkpoint(lightning_module_cfg, ckpt_path):
    module = importlib.import_module(lightning_module_cfg["module"])
    model_class = getattr(module, lightning_module_cfg["main"])
    model = model_class.load_from_checkpoint(ckpt_path, **lightning_module_cfg['args'])
    return model


def download_checkpoint_from_wandb(wandb_id: str, checkpoint_path: str, project: str = "MSR_Separation", 
                                   entity: str = "something_with_audio", artifact_name: str = None) -> str:
    """
    Download checkpoint from wandb artifact if it's not available locally.
    
    Args:
        wandb_id: The wandb run ID
        checkpoint_path: The expected local path where the checkpoint should be
        project: The wandb project name (default: "MSR_Separation")
        entity: The wandb entity name (default: "something_with_audio")
        artifact_name: Optional custom artifact name. If None, uses f"checkpoint-{wandb_id}"
    
    Returns:
        The path to the checkpoint file (either existing or downloaded)
    
    Raises:
        FileNotFoundError: If checkpoint is not found locally and cannot be downloaded from wandb
    """
    import wandb
    
    # Check if checkpoint already exists locally
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    # Check if the checkpoint directory exists (might have been downloaded before)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.exists(checkpoint_dir):
        # Check if there's a checkpoint file in the directory
        checkpoint_filename = os.path.basename(checkpoint_path)
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.ckpt'):
                # Found a checkpoint file, use it
                found_path = os.path.join(checkpoint_dir, file)
                print(f"Found existing checkpoint in cache: {found_path}")
                return found_path
    
    # Try to download from wandb
    print(f"Checkpoint not found locally at {checkpoint_path}")
    print(f"Attempting to download from wandb artifact for run {wandb_id}...")
    
    try:
        # Initialize wandb API (no need to start a run)
        api = wandb.Api()
        
        # Construct artifact name if not provided
        if artifact_name is None:
            artifact_name = f"checkpoint-{wandb_id}"
        
        # Try to get the artifact
        artifact_path = f"{entity}/{project}/{artifact_name}:latest"
        try:
            artifact = api.artifact(artifact_path)
        except Exception as e:
            # Try without :latest tag
            try:
                artifact_path = f"{entity}/{project}/{artifact_name}"
                artifact = api.artifact(artifact_path)
            except Exception as e2:
                raise FileNotFoundError(
                    f"Could not find wandb artifact '{artifact_path}'. "
                    f"Original error: {e}. Alternative error: {e2}"
                )
        
        # Create directory for checkpoint if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Download artifact to a temporary location first
        import tempfile
        temp_download_dir = tempfile.mkdtemp()
        artifact_dir = artifact.download(root=temp_download_dir)
        
        # Find the checkpoint file in the downloaded artifact
        checkpoint_filename = os.path.basename(checkpoint_path)
        downloaded_path = None
        
        # Look for .ckpt files in the artifact directory
        for root, dirs, files in os.walk(artifact_dir):
            for file in files:
                if file.endswith('.ckpt'):
                    candidate_path = os.path.join(root, file)
                    if downloaded_path is None or file == checkpoint_filename:
                        downloaded_path = candidate_path
                        # If the expected filename matches, use it directly
                        if file == checkpoint_filename:
                            break
        
        if downloaded_path is None:
            import shutil
            shutil.rmtree(temp_download_dir, ignore_errors=True)
            raise FileNotFoundError(
                f"Downloaded artifact from wandb but no .ckpt file found in {artifact_dir}"
            )
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Copy to expected location
        import shutil
        shutil.copy2(downloaded_path, checkpoint_path)
        print(f"Copied checkpoint to expected location: {checkpoint_path}")
        
        # Clean up temporary download directory
        shutil.rmtree(temp_download_dir, ignore_errors=True)
        
        print(f"✓ Successfully downloaded checkpoint from wandb artifact: {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path} and could not download from wandb: {e}"
        )