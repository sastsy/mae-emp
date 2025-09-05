#!/usr/bin/env python
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "torch>=1.5.0",
#     "torchvision>=0.6.0",
#     "datasets>=1.8.0",
# ]
# ///

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Any, Union
from enum import Enum

import torch
import torch.nn as nn
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput
try:
    import evaluate
    _acc = evaluate.load("accuracy")
except Exception:
    _acc = None

import torch
from datasets import load_dataset
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ViTImageProcessor,
    ViTMAEConfig,
    ViTMAEForPreTraining,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEForPreTraining,
    ViTMAEForPreTrainingOutput
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import Trainer


""" Pre-training a 🤗 ViT model as an MAE (masked autoencoder), as proposed in https://huggingface.co/papers/2111.06377."""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.56.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")


class BarlowTwinsVariant(Enum):
    PER_BATCH = "per_batch"
    PER_IMAGE = "per_image"
    UNKNOWN = "unknown"
    
    @classmethod
    def _missing_(cls, _):
        return cls.UNKNOWN
    

def is_valid_bt_variant(variant_str: str) -> bool:
    return variant_str in [v.value for v in BarlowTwinsVariant]


@dataclass
class ViTMAEForPreTrainingOutputBT(ViTMAEForPreTrainingOutput):
    
    mae_loss: Optional[torch.FloatTensor] = None
    bt_loss: Optional[torch.FloatTensor] = None


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="cifar10", metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    image_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of the images in the files."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        data_files = {}
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    mask_ratio: float = field(
        default=0.75, metadata={"help": "The ratio of the number of masked tokens in the input sequence."}
    )
    norm_pix_loss: bool = field(
        default=True, metadata={"help": "Whether or not to train with normalized pixel values as target."}
    )
    bt_variant: str = field(
        default="per_image",
        metadata={"help": "Choose Barlow Twins variant: 'per_batch' or 'per_image'."}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(
        default=1e-3, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."}
    )
    # Barlow Twins hyperparameters
    bt_weight: float = field(default=1.0, metadata={"help": "Weight for Barlow Twins loss when added to MAE loss."})
    bt_lambda: float = field(default=5e-3, metadata={"help": "Off-diagonal penalty (lambda) in Barlow Twins loss."})
    bt_eps: float = field(default=1e-9, metadata={"help": "Small eps for std normalization in BT computation."})
    dataloader_num_workers: int = field(
        default=8,
        metadata={"help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."}
    )
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={"help": "Whether to keep dataloader workers alive between epochs."}
    )
    report_to: list[str] = field(
        default_factory=lambda: ["tensorboard"],
        metadata={"help": "The list of integrations to report the results and checkpoints to."}
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    return {"pixel_values": pixel_values}


def collate_fn_cls(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    if _acc is not None:
        return _acc.compute(predictions=preds, references=labels)
    # fallback w/o evaluate
    return {"accuracy": (preds == labels).mean().item()}


class ViTMAETrainer(Trainer):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # --- Logging our custom bt and mae losses ---
        mae_loss = getattr(outputs, "mae_loss", None)
        bt_loss = getattr(outputs, "bt_loss", None)
        
        if self.args.logging_dir and self.state.global_step % self.args.logging_steps == 0:
            logs = {"train/total_loss": loss.item()}
            if mae_loss is not None:
                logs["train/mae_loss"] = mae_loss.item()
            if bt_loss is not None:
                logs["train/bt_loss"] = bt_loss.item()
                
            self.log(logs)
        # --- (end) Logging our custom bt and mae losses ---
        
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


# --- Per-Batch ---
class ViTMAEForPreTrainingWithBT_PerBatch(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.bt_lambda = config.bt_lambda
        self.bt_loss_weight = config.bt_weight
        self.bt_eps = config.bt_eps

    def barlow_twins_loss(self, z1, z2):
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + self.bt_eps)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + self.bt_eps)

        c = torch.mm(z1_norm.T, z2_norm) / z1_norm.shape[0]

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c - torch.eye(c.size(0), device=c.device)).pow_(2).sum()
        return on_diag + self.bt_lambda * off_diag

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[tuple, ViTMAEForPreTrainingOutput]:

        outputs = super().forward(pixel_values=pixel_values, **kwargs)

        logits = outputs.logits # (B, num_patches, patch_size**2 * C)
        mask = outputs.mask # (B, N_patches)

        pred_pixels = logits
        target_pixels = self.patchify(pixel_values)

        visible_mask = (1 - mask).bool()
        real_visible = target_pixels[visible_mask]
        pred_visible = pred_pixels[visible_mask]

        mae_loss = outputs.loss
        bt_loss = self.barlow_twins_loss(
            real_visible.view(real_visible.size(0), -1),
            pred_visible.view(pred_visible.size(0), -1)
        )

        total_loss = mae_loss + self.bt_loss_weight * bt_loss
        return ViTMAEForPreTrainingOutputBT(
            loss=total_loss,
            logits=outputs.logits,
            mask=outputs.mask,
            ids_restore=outputs.ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mae_loss=mae_loss,
            bt_loss=bt_loss,
        )


# --- Per-Image ---
class ViTMAEForPreTrainingWithBT_PerImage(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.bt_lambda = config.get("bt_lambda")
        self.bt_loss_weight = config.get("bt_weight")
        self.bt_eps = config.get("bt_eps")

    def barlow_twins_loss(self, z1, z2):
        if z1.numel() == 0 or z2.numel() == 0:
            return torch.tensor(0.0, device=z1.device)

        if z1.shape[0] < 2:
            return torch.tensor(0.0, device=z1.device)

        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)

        c = torch.mm(z1_norm.T, z2_norm) / z1_norm.shape[0]  # (D, D)

        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()

        off = c.clone()
        off.fill_diagonal_(0.0)
        off_diag = off.pow(2).sum()

        return on_diag + self.bt_lambda * off_diag

    def forward(
        self,
        pixel_values=None,
        noise=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        interpolate_pos_encoding=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore, interpolate_pos_encoding=interpolate_pos_encoding)
        logits = decoder_outputs.logits

        mae_loss = self.forward_loss(pixel_values, logits, mask, interpolate_pos_encoding=interpolate_pos_encoding)

        bt_loss = 0.0
        target_pixels = self.patchify(pixel_values, interpolate_pos_encoding)
        pred_pixels = logits

        B = target_pixels.size(0)
        bt_loss_accum = 0.0
        for b in range(B):
            visible_mask = (mask[b] == 0)
            if visible_mask.sum() == 0:
                continue
            real_b = target_pixels[b][visible_mask]
            pred_b = pred_pixels[b][visible_mask]
            bt_loss_accum += self.barlow_twins_loss(real_b, pred_b)

        bt_loss = bt_loss_accum / B
        total_loss = mae_loss + self.bt_loss_weight * bt_loss

        if not return_dict:
            return (total_loss, logits, mask, ids_restore)

        return ViTMAEForPreTrainingOutputBT(
            loss=total_loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mae_loss=mae_loss,
            bt_loss=bt_loss,
        )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mae", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our dataset.
    ds = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        data_files=data_args.data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in ds else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = ds["train"].train_test_split(data_args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    # Load pretrained model and image processor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
    }
    if model_args.config_name:
        config = ViTMAEConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = ViTMAEConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = ViTMAEConfig()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # adapt config
    config.update(
        {
            "mask_ratio": model_args.mask_ratio,
            "norm_pix_loss": model_args.norm_pix_loss,
        }
    )
    if is_valid_bt_variant(model_args.bt_variant):
        config.update(
            {
                "bt_weight": training_args.bt_weight,
                "bt_lambda": training_args.bt_lambda,
                "bt_eps": training_args.bt_eps,
            }
        )

    # create image processor
    if model_args.image_processor_name:
        image_processor = ViTImageProcessor.from_pretrained(model_args.image_processor_name, **config_kwargs)
    elif model_args.model_name_or_path:
        image_processor = ViTImageProcessor.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        image_processor = ViTImageProcessor()
    
    # if we want to train with barlow-twins variants
    if BarlowTwinsVariant(model_args.bt_variant) is BarlowTwinsVariant.PER_BATCH:
        ModelCls = ViTMAEForPreTrainingWithBT_PerBatch 
    elif BarlowTwinsVariant(model_args.bt_variant) is BarlowTwinsVariant.PER_IMAGE:
        ModelCls = ViTMAEForPreTrainingWithBT_PerImage 
    else:
        print(f"Unknown bt_variant: {model_args.bt_variant}, falling back to ViTMAEForPreTraining")
        ModelCls = ViTMAEForPreTraining

    # create model
    if model_args.model_name_or_path:
        model = ModelCls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
        )
    else:
        logger.info("Training new model from scratch")
        model = ModelCls(config)

    if training_args.do_train:
        column_names = ds["train"].column_names
    else:
        column_names = ds["validation"].column_names

    if data_args.image_column_name is not None:
        image_column_name = data_args.image_column_name
    elif "image" in column_names:
        image_column_name = "image"
    elif "img" in column_names:
        image_column_name = "img"
    else:
        image_column_name = column_names[0]

    # transformations as done in original MAE paper
    # source: https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    transforms = Compose(
        [
            Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )

    def preprocess_images(examples):
        """Preprocess a batch of images by applying transforms."""

        examples["pixel_values"] = [transforms(image) for image in examples[image_column_name]]
        return examples

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        # Set the training transforms
        ds["train"].set_transform(preprocess_images)

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = (
                ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        ds["validation"].set_transform(preprocess_images)

    # Compute absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    # Initialize our trainer
    trainer = ViTMAETrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "tasks": "masked-auto-encoding",
        "dataset": data_args.dataset_name,
        "tags": ["masked-auto-encoding"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()