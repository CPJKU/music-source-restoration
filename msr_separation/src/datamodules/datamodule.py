import sofa
from typing import Dict, List, Optional, NoReturn
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import importlib    

def initialize_config(module_cfg):
    module = importlib.import_module(module_cfg["module"])
    if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else: return getattr(module, module_cfg["main"])()

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataloader: dict,
        val_dataloader: dict = None,
        test_dataloader: dict = None,
    ):
        super().__init__()
        self.train_config = train_dataloader
        self.val_config = val_dataloader
        self.test_config = test_dataloader

        self.train_dataset = initialize_config(self.train_config['dataset'])
        if val_dataloader is not None:            
            self.val_dataset = initialize_config(self.val_config['dataset'])
        else:
            self.val_dataset = None
        
        if test_dataloader is not None:
            self.test_dataset = initialize_config(self.test_config['dataset'])
        else:
            self.test_dataset = None


    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader."""
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_config['batch_size'],
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.train_config['num_workers'],
            pin_memory=True,
            persistent_workers=self.train_config['persistent_workers'],
            shuffle=True
        )

        return train_loader

    def val_dataloader(self):
        if self.val_dataset is not None:
            val_dataloder = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.val_config['batch_size'],
                collate_fn=self.val_dataset.collate_fn,
                num_workers=self.val_config['num_workers'],
                pin_memory=True,
                persistent_workers=self.val_config['persistent_workers'],
                shuffle=False
            )
            return val_dataloder

    def test_dataloader(self):
        # Use explicit test_dataloader config if provided
        if self.test_config is not None:
            test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.test_config['batch_size'],
                collate_fn=self.test_dataset.collate_fn,
                num_workers=self.test_config['num_workers'],
                pin_memory=True,
                persistent_workers=self.test_config.get('persistent_workers', False),
                shuffle=False
            )
            return test_dataloader

    def teardown(self, stage=None):
        pass
