import os
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any


@dataclass
class Config:

    # Paths (these get overridden by main_net.py from --data-path/--save-dir)
    data_path: str = "/Users/rosalouisemarker/Desktop/Digital Media Project/dataset"
    save_dir: str = "net/save_dir"


    # Data / modalities
    dataset: str = "SZ2"                  # used for SZ2_training/validation/test.tsv
    modalities: List[str] = field(default_factory=lambda: ["eeg", "ecg"])
    # input_mode in {"eeg", "hrv", "eeg_hrv"}; main_net overrides based on --exp
    input_mode: str = "eeg_hrv"


    # Segmentation / sampling
    fs: int = 250                        
    CH: int = 2                         
    cross_validation: str = "fixed"     
    frame: float = 2.0                   
    stride: float = 1.0                   
    stride_s: float = 0.5                 
    boundary: float = 0.5                 
    factor: int = 5                       
    sample_type: str = "subsample"        

   
    # Model / training hyperparameters
    model: str = "ChronoNet"            
    nb_epochs: int = 10
    batch_size: int = 128
    dropoutRate: float = 0.5
    l2: float = 0.01
    lr: float = 1e-3                   
  
    # Classification specifics
    # binary: 0 = inter-ictal/ictal, 1 = pre-ictal 
    num_classes: int = 2
    class_weights: Dict[int, float] = field(default_factory=lambda: {0: 1.0, 1: 2.0})

    # Naming / bookkeeping
    add_to_name: str = ""                

    def get_name(self) -> str:
        base = f"{self.dataset}_{self.model}_frame-{self.frame}_mode-{self.input_mode}"
        if self.add_to_name:
            base = f"{base}_{self.add_to_name}"
        return base

    @property
    def name(self) -> str: 
        return self.get_name()

    # Config persistence (train: save_config, predict: load_config)
    def save_config(self, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)
        cfg_file = os.path.join(save_path, self.get_name() + ".cfg")
        with open(cfg_file, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def load_config(self, config_path: str, config_name: str) -> None:
        full_path = os.path.join(config_path, config_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Config file not found: {full_path}")
        with open(full_path, "r") as f:
            data: Dict[str, Any] = json.load(f)

        # Update fields in-place; ignore unknown keys
        for k, v in data.items():
            setattr(self, k, v)

    def apply_experiment_profile(self, exp: str) -> None:
        exp = exp.lower()

        if exp == "eeg":
            self.input_mode = "eeg"
            self.modalities = ["eeg"]
            self.add_to_name = "eeg"

        elif exp == "hrv":
            # HRV derived from ECG
            self.input_mode = "hrv"
            self.modalities = ["ecg"]
            self.add_to_name = "hrv"

        elif exp == "eeg_hrv":
            self.input_mode = "eeg_hrv"
            self.modalities = ["eeg", "ecg"]
            self.add_to_name = "eeg_hrv"

        else:
            self.add_to_name = exp

DL_Config = Config