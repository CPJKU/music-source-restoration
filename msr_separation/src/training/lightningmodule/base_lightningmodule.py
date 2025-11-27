from typing import Any, Callable, Dict
import lightning.pytorch as pl
import torch
from huggingface_hub import PyTorchModelHubMixin
import transformers
import math

from src.utils import initialize_config

def get_lr_scheduler(
        optimizer,
        num_training_steps,
        schedule_mode="exp",
        gamma: float = 0.999996,  # Default for exp scheduler. For step_lr, use 0.9 (specify in config)
        num_warmup_steps=20000,
        lr_end=2e-7,
        step_size=4000,
        # Cyclic LR parameters
        base_lr=None,
        max_lr=None,
        step_size_up=90,  # Default: 90 steps = ~15 epochs (for 6 steps/epoch), gives ~4 cycles over 120 epochs
        step_size_down=None,
        mode='triangular',
        cyclic_gamma=1.0,
        scale_mode='cycle',
):
    if schedule_mode in {"exp"}:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    if schedule_mode in {"cosine", "cos"}:
        # Check if optimizer has multiple parameter groups with different learning rates
        # Normalize learning rates to floats in case they're sequences
        initial_lrs = []
        for group in optimizer.param_groups:
            lr = group['lr']
            if isinstance(lr, (list, tuple)):
                if len(lr) > 0:
                    lr = float(lr[0])
                else:
                    raise ValueError(f"Learning rate sequence is empty: {lr}")
            else:
                lr = float(lr)
            initial_lrs.append(lr)
        has_different_lrs = len(set(initial_lrs)) > 1
        
        if has_different_lrs:
            # Custom cosine scheduler for multiple parameter groups with different initial LRs
            # LambdaLR will apply the scale factor to each parameter group's current lr,
            # preserving the relative learning rates between groups
            
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, num_warmup_steps))
                else:
                    # Cosine decay
                    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            # Use the same lambda for all parameter groups - LambdaLR will apply it to each group's current lr
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # Standard transformers scheduler works fine when all groups have the same LR
            return transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
    if schedule_mode in {"linear"}:
        print("Linear schedule!")
        return transformers.get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=1.0,
            lr_end=lr_end,
        )
    if schedule_mode in {"step_lr", "step"}:
        print(f"Step LR schedule with step_size={step_size} and gamma={gamma}")
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    if schedule_mode in {"cyclic", "cyclic_lr"}:
        # Get initial learning rate from optimizer if max_lr not specified
        if max_lr is None:
            # Use the first parameter group's lr as max_lr
            max_lr = optimizer.param_groups[0]['lr']
            # Ensure max_lr is a float, not a list/tuple
            if isinstance(max_lr, (list, tuple)):
                if len(max_lr) > 0:
                    max_lr = float(max_lr[0])
                else:
                    raise ValueError(f"Learning rate sequence is empty: {max_lr}")
            else:
                max_lr = float(max_lr)
        
        # Set base_lr if not specified (default to lr_end or a fraction of max_lr)
        if base_lr is None:
            if lr_end is not None and lr_end > 0:
                base_lr = lr_end
            else:
                # Default to 1/10 of max_lr if lr_end not specified
                base_lr = max_lr / 10.0
        
        # Set step_size_down if not specified (defaults to step_size_up)
        if step_size_down is None:
            step_size_down = step_size_up
        
        print(f"Cyclic LR schedule:")
        print(f"  base_lr: {base_lr}")
        print(f"  max_lr: {max_lr}")
        print(f"  step_size_up: {step_size_up}")
        print(f"  step_size_down: {step_size_down}")
        print(f"  mode: {mode}")
        if mode == 'exp_range':
            print(f"  gamma: {cyclic_gamma}")
        print(f"  scale_mode: {scale_mode}")
        
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            mode=mode,
            gamma=cyclic_gamma,
            scale_mode=scale_mode,
        )
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")


class BaseLightningModule(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        model: Dict,
        loss: Dict,
        optimizer: Dict,
        lr_scheduler:Dict=None,
        is_validation=False,
        metric:Dict=None,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
    ):

        super().__init__()
        self.model_config = model
        self.model = initialize_config(self.model_config)

        self.loss_config = loss
        self.loss_func = initialize_config(self.loss_config)

        self.optimizer_config = optimizer
        if self.optimizer_config.get('split_params', False):
            if hasattr(self.model, 'separate_params'):
                lr_dprnn_in_config = 'lr_dprnn' in self.optimizer_config
                if lr_dprnn_in_config:
                    raise NotImplementedError()
                    lr_dprnn = self.optimizer_config['lr_dprnn'] \
                        if self.optimizer_config['lr_dprnn'] is not None \
                        else self.optimizer_config['args']['lr']

                    params = self.model.separate_params(
                        self.optimizer_config['args']['lr'],
                        lr_dprnn,
                        self.optimizer_config.get('lr_decay_factor', 1.0),
                        self.optimizer_config['args'].get('weight_decay', 0.0),
                    )
                else:
                    params = self.model.separate_params(
                        self.optimizer_config['args']['lr'],
                        self.optimizer_config['lr_mask_pretrained'],
                        self.optimizer_config['lr_mask_new'],
                        self.optimizer_config.get('lr_decay_factor', 1.0),
                        self.optimizer_config['args'].get('weight_decay', 0.0),
                    )
            else:
                raise NotImplementedError(f"Model {type(self.model)} does not implement `separate_params(...)`.")
        else:
            params = self.model.parameters()

        self.optimizer_config['args']['params'] = params  # modify if some parts are frozen

        self.optimizer = initialize_config(self.optimizer_config)

        self.lr_scheduler_config = lr_scheduler

        if is_validation:
            if metric:
                self.metric_config = metric
                self.metric_func = initialize_config(self.metric_config)
            else:
                self.metric_func = None
        
        self.is_validation = is_validation

        # Initialize EMA if enabled
        self.use_ema = use_ema
        if self.use_ema:
            from .ema import ExponentialMovingAverage
            self.ema = ExponentialMovingAverage(self.model, decay=ema_decay)
            print(f"EMA enabled with decay={ema_decay}")
        else:
            self.ema = None

    def _normalize_learning_rates(self):
        """Ensure all learning rates in parameter groups are floats (not lists/tuples).
        
        This can happen when resuming from checkpoints that were saved incorrectly
        or when learning rates are set as sequences instead of scalars.
        """
        for group in self.optimizer.param_groups:
            if 'lr' in group:
                lr = group['lr']
                if isinstance(lr, (list, tuple)):
                    # If lr is a sequence, take the first element
                    if len(lr) > 0:
                        group['lr'] = float(lr[0])
                        print(f"WARNING: Learning rate was a sequence {lr}, converted to float: {group['lr']}")
                    else:
                        raise ValueError(f"Learning rate sequence is empty: {lr}")
                elif not isinstance(lr, (int, float)):
                    # Convert to float if it's some other numeric type
                    group['lr'] = float(lr)

    def forward(self, x):
        raise NotImplementedError()

    def training_step(self, batch_data_dict, batch_idx):
        raise NotImplementedError()


    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step == self.optimizer_config["unfreeze_mask_at_step"]:
            group_mask = next((g for g in self.optimizer.param_groups if g.get('name') == 'mask_pretrained'), None)
            lr_value = self.optimizer_config["lr_mask_pretrained_unlocked"]
            # Ensure learning rate is a float, not a list/tuple
            if isinstance(lr_value, (list, tuple)):
                if len(lr_value) > 0:
                    lr_value = float(lr_value[0])
                    print(f"WARNING: lr_mask_pretrained_unlocked was a sequence, converted to float: {lr_value}")
                else:
                    raise ValueError(f"lr_mask_pretrained_unlocked sequence is empty: {lr_value}")
            else:
                lr_value = float(lr_value)
            group_mask['lr'] = lr_value

            print(f"On_train_batch_start: at step {self.global_step} the lr for parametergroup 'mask_pretrained'" \
                  + f"is set to {group_mask['lr']}")

    def on_train_epoch_end(self):
        raise NotImplementedError()

    def validation_step(self, batch_data_dict, batch_idx):
        raise NotImplementedError()

    def on_validation_epoch_end(self):
        raise NotImplementedError()


    # def training_step_processing(self, batch_data_dict, batch_idx):
    #     raise NotImplementedError
    #
    #     batchsize = batch_data_dict['mixture'].shape[0]
    #
    #     input_dict = {
    #         'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
    #         'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
    #         }
    #     output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}
    #     target_dict = {'waveform': batch_data_dict['ground_truth']}
    #     loss_dict = self.loss_func(output_dict, target_dict)
    #
    #     return batchsize, loss_dict

    # def training_step(self, batch_data_dict, batch_idx):
    #     self.set_train_mode()
    #
    #     batchsize, loss_dict = self.training_step_processing(batch_data_dict, batch_idx)
    #
    #     loss = loss_dict['loss'] # for back propagation
    #
        # # log all items in loss_dict
        # step_dict = {f'step_train/{name}': val.item() if hasattr(val, "item") else val for name, val in loss_dict.items()}
        # self.log_dict(step_dict, prog_bar=False, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)
        # epoc_dict = {f'epoch_train/{name}': val.item() if hasattr(val, "item") else val for name, val in loss_dict.items()}
        # self.log_dict(epoc_dict, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)
        #
        # self.log_dict({"epoch/lr": self.optimizer.param_groups[0]['lr']},)  # this here results in a warning
    #
    #     return loss

    # def on_train_epoch_end(self):
    #     raise NotImplementedError


    # def validation_step_processing(self, batch_data_dict, batch_idx):
    #     raise NotImplementedError
    #     # batchsize = batch_data_dict['mixture'].shape[0]
    #
    #     # input_dict = {
    #     #     'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
    #     #     'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
    #     #     }
    #     # output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}
    #     # target_dict = {'waveform': batch_data_dict['ground_truth']}
    #     # loss_dict = self.loss_func(output_dict, target_dict)
    #
    #     # loss_dict = {k: v.item() for k,v in loss_dict.items()}
    #     # if self.metric_func: # add metrics
    #     #     metric = self.metric_func(output_dict, target_dict)
    #     #     for k,v in metric.items():
    #     #         loss_dict[k] = v.mean().item() # torch tensor size [bs]
    #
    #     # return batchsize, loss_dict

    # def _validation_step(self, batch_data_dict, batch_idx):
    #     self.model.eval()
    #
    #     batchsize, loss_dict = self.validation_step_processing(batch_data_dict, batch_idx)
    #
        # # log all items in loss_dict
        # step_dict = {f'step_val/{name}': metric for name, metric in loss_dict.items()}
        # self.log_dict(step_dict, prog_bar=False, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)
        # epoc_dict = {f'epoch_val/{name}': metric for name, metric in loss_dict.items()}
        # self.log_dict(epoc_dict, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)
    #
    # def on_validation_epoch_end(self):
    #     raise NotImplementedError

    def configure_optimizers(self):
        r"""Configure optimizer.
            will be called automatically
        """
        # Ensure all learning rates in parameter groups are floats (not lists/tuples)
        # This can happen when resuming from checkpoints that were saved incorrectly
        self._normalize_learning_rates()
        
        if self.lr_scheduler_config and self.lr_scheduler_config['schedule_mode'] is not None:
            # Get estimated stepping batches - this might be None during initialization
            # PyTorch Lightning will restore scheduler state from checkpoint when resuming
            num_training_steps = self.trainer.estimated_stepping_batches
            
            # If estimated_stepping_batches is None (can happen during initialization),
            # use a placeholder value. PyTorch Lightning will restore the actual scheduler
            # state from checkpoint when resuming, so this is only used for fresh runs.
            if num_training_steps is None:
                # Estimate based on max_epochs if available
                max_epochs = getattr(self.trainer, 'max_epochs', None)
                if max_epochs is not None:
                    # Rough estimate: assume ~6 steps per epoch (will be updated when trainer is ready)
                    num_training_steps = max_epochs * 6
                    print(f"WARNING: estimated_stepping_batches is None, using placeholder: {num_training_steps}")
                else:
                    num_training_steps = 120*6  # Fallback placeholder
                    print(f"WARNING: estimated_stepping_batches is None and max_epochs unavailable, using placeholder: {num_training_steps}")

            schedule_mode = self.lr_scheduler_config['schedule_mode']
            num_warmup_steps = self.lr_scheduler_config.get('num_warmup_steps', 0)
            lr_end = self.lr_scheduler_config.get('lr_end', None)
            
            # Extract step_size and gamma for step_lr scheduler
            # Default gamma for step_lr is 0.9 (reduce LR by 10% each step)
            step_size = self.lr_scheduler_config.get('step_size', 4000)
            gamma = self.lr_scheduler_config.get('gamma', 0.9)
            
            # Extract cyclic LR parameters
            # Default step_size_up is 90 steps (~15 epochs for 6 steps/epoch), giving ~4 cycles over 120 epochs
            base_lr = self.lr_scheduler_config.get('base_lr', None)
            max_lr = self.lr_scheduler_config.get('max_lr', None)
            step_size_up = self.lr_scheduler_config.get('step_size_up', 90)
            step_size_down = self.lr_scheduler_config.get('step_size_down', None)
            cyclic_mode = self.lr_scheduler_config.get('mode', 'triangular')
            cyclic_gamma = self.lr_scheduler_config.get('cyclic_gamma', 1.0)
            scale_mode = self.lr_scheduler_config.get('scale_mode', 'cycle')

            print("schedule_mode: ", schedule_mode)
            print("num_warmup_steps: ", num_warmup_steps)
            print("lr_end: ", lr_end)
            print("num_training_steps: ", num_training_steps)
            if schedule_mode in {"step_lr", "step"}:
                print("step_size: ", step_size)
                print("gamma: ", gamma)
            if schedule_mode in {"cyclic", "cyclic_lr"}:
                print("base_lr: ", base_lr)
                print("max_lr: ", max_lr)
                print("step_size_up: ", step_size_up)
                print("step_size_down: ", step_size_down)
                print("mode: ", cyclic_mode)
                print("cyclic_gamma: ", cyclic_gamma)
                print("scale_mode: ", scale_mode)

            scheduler = get_lr_scheduler(
                self.optimizer,
                num_training_steps,
                schedule_mode=schedule_mode,
                num_warmup_steps=num_warmup_steps,
                lr_end=lr_end,
                step_size=step_size,
                gamma=gamma,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                mode=cyclic_mode,
                cyclic_gamma=cyclic_gamma,
                scale_mode=scale_mode,
            )

            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    'scheduler': scheduler,
                    'interval': "step",
                    'frequency': 1,
                }
            }
        else:
            return self.optimizer

    def setup(self, stage):
        """
        Called at the beginning of fit (train + validate), validate, test, or predict.
        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using DDP.
        """
        # Normalize learning rates after checkpoint loading (this runs after state restoration)
        # This ensures learning rates are floats even if checkpoint restored them as sequences
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self._normalize_learning_rates()
        
        # Reinitialize EMA after model has been moved to the correct device
        # Only if EMA state wasn't loaded from checkpoint (on_load_checkpoint handles that)
        if self.use_ema and self.ema is not None:
            # Check if EMA was already initialized (from checkpoint)
            if not self.ema.ema_params:
                # Reinitialize EMA parameters on the correct device
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.ema.ema_params[name] = param.data.clone().to(param.device)
            else:
                # EMA was loaded from checkpoint, just ensure it's on the correct device
                for name, param in self.model.named_parameters():
                    if name in self.ema.ema_params:
                        self.ema.ema_params[name] = self.ema.ema_params[name].to(param.device)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Perform a single optimization step.
        
        Override to add custom behavior like gradient clipping or EMA updates.
        """
        # Call the actual optimizer step
        optimizer.step(closure=optimizer_closure)
        
        # Update EMA after optimizer step
        if self.use_ema and self.ema is not None:
            self.ema.update()

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        pass

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        pass

    def on_save_checkpoint(self, checkpoint):
        """Save EMA state to checkpoint."""
        if self.use_ema and self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        """Load EMA state from checkpoint."""
        if self.use_ema and self.ema is not None:
            if 'ema_state_dict' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
                print("Loaded EMA state from checkpoint")
            else:
                print("Warning: EMA enabled but no EMA state found in checkpoint. EMA will be reinitialized.")