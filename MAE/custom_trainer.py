from tqdm import tqdm
from typing import Union, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def collate_fn_cls(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}


class LinearProbe:
    def __init__(self, input_dim, num_classes, device="cuda", lr=1e-3, epochs=1):
        self.device = device
        self.epochs = epochs
        self.linear = nn.Linear(input_dim, num_classes).to(device)
        self.optimizer = optim.Adam(self.linear.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def fit(self, features, labels):
        self.linear.train()
        dataset = torch.utils.data.TensorDataset(features.cpu(), labels.cpu())
        loader = DataLoader(
            dataset,
            batch_size=64,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
        )

        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.linear(x)
                loss = self.loss_fn(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, features):
        self.linear.eval()
        with torch.no_grad():
            features = features.to(self.device)
            logits = self.linear(features)
            return logits.argmax(dim=-1)

    def evaluate(self, features, labels):
        labels = labels.to(self.device)
        preds = self.predict(features)
        acc = (preds == labels).float().mean().item()
        return acc


class ViTMAETrainer(Trainer):
    def __init__(self, *args, num_classes=1000, probe_lr=1e-3, probe_epochs=1, do_probe=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_probe = do_probe
        
        if self.do_probe:
            hidden_size = self.model.config.hidden_size
            self.linear_probe = LinearProbe(
                input_dim=hidden_size,
                num_classes=num_classes,
                device=self.args.device,
                lr=probe_lr,
                epochs=probe_epochs,
            )
    
    @torch.no_grad()
    def extract_features_dataset(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn_cls
        )
        
        all_features = []
        all_labels = []
        for batch in tqdm(dataloader, desc="Extracting features"):
            x = batch["pixel_values"].to(self.args.device)
            z = self.model.vit(x).last_hidden_state
            img_repr = z.mean(dim=1)  # global average pooling
            
            all_features.append(img_repr)
            all_labels.append(batch["labels"])
            
        return torch.cat(all_features), torch.cat(all_labels)

    def evaluate_with_probe(self, eval_dataset=None):
        eval_dataset = eval_dataset or self.eval_dataset
        features, labels = self.extract_features_dataset(eval_dataset)

        # Train linear probe
        self.linear_probe.fit(features, labels)
        acc = self.linear_probe.evaluate(features, labels)
        
        metrics = {"eval/linear_probe_accuracy": acc}
        self.log(metrics)
        
        return metrics

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        
        if self.do_probe:
            probe_metrics = self.evaluate_with_probe()
            metrics.update(probe_metrics)
        
        return metrics
        
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
            if mae_loss is not None and bt_loss is not None:
                self.log({
                    "mae_loss": mae_loss.item(),
                    "bt_loss": bt_loss.item(),
                })
        # --- (end) Logging our custom bt and mae losses ---
        
        if not model.training and mae_loss is not None:
            loss = mae_loss
        
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
