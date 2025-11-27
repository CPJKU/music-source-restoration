from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from .configs import DataConfig, ModelConfig, StageConfig, TrainConfig
from .trainer import FinallyGanTrainer


def run_training(
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    data_cfg: DataConfig | None = None,
    resume_from: str | None = None,
) -> None:
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()
    data_cfg = data_cfg or DataConfig()
    trainer = FinallyGanTrainer(
        model_cfg=model_cfg, 
        train_cfg=train_cfg, 
        data_cfg=data_cfg,
        resume_from=resume_from
    )
    trainer.train()


def dataclass_from_json(dataclass_type, path: str):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if dataclass_type is TrainConfig:
        stages = payload.get("stages")
        if stages is not None:
            payload["stages"] = [StageConfig(**stage) for stage in stages]
    return dataclass_type(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Finally-style GAN for music restoration.")
    parser.add_argument("--model-cfg", type=str, default=None, help="Path to JSON file overriding ModelConfig.")
    parser.add_argument("--train-cfg", type=str, default=None, help="Path to JSON file overriding TrainConfig.")
    parser.add_argument("--data-cfg", type=str, default=None, help="Path to JSON file overriding DataConfig.")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint file to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = dataclass_from_json(ModelConfig, args.model_cfg) if args.model_cfg else ModelConfig()
    train_cfg = dataclass_from_json(TrainConfig, args.train_cfg) if args.train_cfg else TrainConfig()
    data_cfg = dataclass_from_json(DataConfig, args.data_cfg) if args.data_cfg else DataConfig()
    run_training(model_cfg=model_cfg, train_cfg=train_cfg, data_cfg=data_cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
