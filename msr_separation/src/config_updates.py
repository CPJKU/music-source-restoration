import os
import socket

hostname = socket.gethostname()

RESOURCES_FOLDER = "resources"

IR_PATHS = "/opt/datasets/music-enhancement/EchoThiefImpulseResponseLibrary/"
DATA_PATH_4_SOURCES = "/opt/datasets/HF_datasets/saved/mss4s_combined"
DATA_PATH_8_SOURCES = "/opt/datasets/HF_datasets/saved/mss8s_other_is_present"
DATA_PATH_MSR_BENCH = "/opt/datasets/HF_datasets/saved/msr_bench"

MUSICA_SCRATCH_4_STEM = "/scratch/<username>/msr_separation/4_stem"
MUSICA_SCRATCH_8_STEM = "/scratch/<username>/msr_separation/8_stem"

if "rk" in hostname or "ed" in hostname: # RKs
    username = os.getlogin()

    LOG_PATH = f"/var/home/{username}/msr_separation/logs"
    CKPT_PATH = f"/var/home/{username}/msr_separation/checkpoints"
elif hostname.startswith('n'):
    print("MUSICA cluster detected")

    LOG_PATH = f"/scratch/<username>/msr_separation/wandb/logs"
    CKPT_PATH = f"/data/<username>/msr_checkpoints"

else:
    LOG_PATH = os.path.join("outputs", "logs")
    CKPT_PATH = os.path.join("outputs", "checkpoints")


def add_configs(ex):
    """
    This functions add generic configuration for the experiments, such as mix-up, architectures, etc...
    """
    @ex.named_config
    def roformer_4s_dataset():
        config_yaml_file = "config/roformer_4s-dataset.yaml"

    @ex.named_config
    def roformer_4s_dataset_step_lr_scheduler():
        config_yaml_file = "config/roformer_4s-dataset_step_lr_scheduler.yaml"

    @ex.named_config
    def roformer_4s_dataset_fine_tuning():
        config_yaml_file = "config/roformer_4s-dataset_fine-tuning.yaml"

    @ex.named_config
    def roformer_4s_dataset_mixed():
        config_yaml_file = "config/roformer_4s-dataset-mixed.yaml"

    @ex.named_config
    def roformer_8s_dataset():
        config_yaml_file = "config/roformer_8s-dataset.yaml"

    @ex.named_config
    def roformer_8s_dataset_evaluate():
        config_yaml_file = "config/roformer_8s-dataset_evaluate.yaml"

    @ex.named_config
    def roformer_8s_dataset_generate():
        config_yaml_file = "config/roformer_8s-dataset_generate.yaml"

    @ex.named_config
    def roformer_8s_dataset_test_inference():
        config_yaml_file = "config/roformer_8s-dataset_test_inference.yaml"

    @ex.named_config
    def only_validate_and_upload():
        config_yaml_file = "config/roformer_8s-dataset_evaluate.yaml"
        validate_and_upload_only = True

    @ex.named_config
    def roformer_8s_dataset_cos():
        config_yaml_file = "config/roformer_8s-dataset_cos.yaml"

    @ex.named_config
    def roformer_8s_dataset_msr_bench_val():
        config_yaml_file = "config/roformer_8s-dataset_msr-bench-val.yaml"


    @ex.named_config
    def roformer_4s_dataset_small():
        config_yaml_file = "config/roformer_4s-dataset_small.yaml"

    @ex.named_config
    def musica():
        train = dict(
            trainer = dict(
                args = dict(
                    # I don't know if it was good, but before we had ACC=32 and BS=2
                    # now we have BS=2 * 4 GPUs = 8 effective batch size --> acc = 2 or 4
                    accumulate_grad_batches = 2  # 4
                )
            )
        )

    @ex.named_config
    def only_finetune_new_masks_3e_3():
        lightning_module = dict(
            args = dict(
                model = dict(
                    args = dict(
                        lora_config = dict(
                            finetune_lora = 0,
                            finetune_mask_estimator = 1,
                            finetune_only_new_mask_estimators = 1
                        )
                    )
                ),
                optimizer = dict(
                    lr_mask_pretrained = 0,
                    lr_mask_pretrained_unlocked = 0,
                    lr_mask_new = 3e-3
                )
            )
        )

    @ex.named_config
    def only_finetune_new_masks():
        lightning_module = dict(
            args = dict(
                model = dict(
                    args = dict(
                        lora_config = dict(
                            finetune_lora = 0,
                            finetune_mask_estimator = 1,
                            finetune_only_new_mask_estimators = 1
                        )
                    )
                ),
                optimizer = dict(
                    lr_mask_pretrained = 0,
                    lr_mask_pretrained_unlocked = 0,
                )
            )
        )

    @ex.named_config
    def small_gpu():
        datamodule = dict(
            args = dict(
                train_dataloader = dict(
                    batch_size=1
                ),
                val_dataloader = dict(
                    batch_size=1
                )
            )
        )

    @ex.named_config
    def mix_8stem_training():
        datamodule = dict(
            args = dict(
                train_dataloader = dict(
                    dataset = dict(
                        main = "MultiAudioRandomSegmentMixDataset",
                        args = dict(
                            mix_probability_start = 1.0,
                            mix_probability_end = 1.0
                        )
                    )
                )
            )
        )


    @ex.named_config
    def cyclic_LR_scheduler():
        lightning_module = dict(
            args = dict(
                lr_scheduler = dict(
                    schedule_mode = 'cyclic',
                    base_lr = 1e-4,
                    max_lr = 8e-3,
                    step_size_up = 200,  # number of training steps to reach max_lr
                    step_size_down = 200,  # number of training steps to go back to base_lr
                    mode = 'triangular',
                    cyclic_gamma = 1.0,
                    scale_mode = "cycle",
                    interval='step',
                    frequency=1
                )
            )
        )


    @ex.named_config
    def aug_sep():
        lightning_module = dict(
            args = dict(
                augment=True,
                target_key="aug_waveform"
            )
        )


    @ex.named_config
    def dprnn():
        lightning_module = dict(
            args = dict(
                model = dict(
                    args = dict(
                        dprnn = dict(
                            use = True,
                        )
                    )
                )
            )
        )

    @ex.named_config
    def iter2():
        lightning_module = dict(
            args = dict(
                model = dict(
                    args = dict(
                        iterative_refinement = dict(
                            use = True,
                            feedback_channels = 8,  # 4 sources x 2 channels (stereo)
                            training_iters = 2,
                            inference_iters = 2,
                            detach_feedback = True,
                        )
                    )
                )
            )
        )

        # you probably need smaller batch-sizes
        datamodule = dict(
            args = dict(
                train_dataloader = dict(
                    batch_size = 2
                ),
                val_dataloader = dict(
                    batch_size = 2
                )
            )
        )


    # @ex.named_config
    # def cosine_lr_scheduler():
    #     lightning_module = dict(
    #         args = dict(
    #             lr_scheduler = dict(
    #                 schedule_mode = 'cos',
    #                 num_warmup_steps = 2_000, # 10% of the total steps when running for 20_000 steps (200 epochs)
    #                 lr_end = 1e-6,
    #                 interval='step',
    #                 frequency=1
    #             )
    #         )
    #     )

    @ex.named_config
    def masking_p08():
        lightning_module = dict(
            args=dict(
                loss=dict(
                    args=dict(
                        masking_p=0.8
                    )
                )
            )
        )


    @ex.named_config
    def layerwise_lr_decay():
        lightning_module = dict(
            args = dict(
                optimizer = dict(
                    split_params=True,
                    lr_decay_factor=0.8,
                )
            )
        )

    @ex.named_config
    def batch_size_2():
        datamodule = dict(
            args = dict(
                train_dataloader = dict(
                    batch_size = 2
                ),
                val_dataloader = dict(
                    batch_size = 2
                )
            )
        )

        train = dict(
            trainer = dict(
                args = dict(
                    accumulate_grad_batches = 2
                )
            )
        )

    @ex.named_config
    def dec_step_lr_scheduler():
        lightning_module = dict(
            args = dict(
                lr_scheduler = dict(
                    schedule_mode = 'step_lr',
                    num_warmup_steps=80,  # not used in step_lr
                    lr_end=2.0e-6,  # not used in step_lr
                    step_size = 200,  # reduce LR every 200 steps
                    gamma = 0.9,
                    interval='step',
                    frequency=1
                )
            )
        )
