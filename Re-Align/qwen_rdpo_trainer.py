import inspect
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_callback import TrainerCallback

class rDPOTrainer(Trainer):
    def __init__(
        self,
        model,
        ref_model,
        args: TrainingArguments,
        beta: float,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        
        # Move reference model to same device as model
        if hasattr(self.model, "module"):
            self.ref_model = self.ref_model.to(self.model.module.device)
        else:
            self.ref_model = self.ref_model.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass through model
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get chosen and rejected sequences
        batch_size = inputs["chosen_input_ids"].shape[0]
        seq_len = inputs["chosen_input_ids"].shape[1]
        
        chosen_ids = inputs["chosen_input_ids"]
        rejected_ids = inputs["rejected_input_ids"]
        
        chosen_logits = logits[:batch_size, :seq_len, :]
        rejected_logits = logits[batch_size:, :seq_len, :]
        
        # Calculate log probabilities
        chosen_log_probs = self._get_batch_logps(
            chosen_logits,
            chosen_ids,
            average_log_prob=False,
        )
        rejected_log_probs = self._get_batch_logps(
            rejected_logits, 
            rejected_ids,
            average_log_prob=False,
        )
        
        # Get reference model outputs
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits
            
            ref_chosen_logits = ref_logits[:batch_size, :seq_len, :]
            ref_rejected_logits = ref_logits[batch_size:, :seq_len, :]
            
            ref_chosen_log_probs = self._get_batch_logps(
                ref_chosen_logits,
                chosen_ids,
                average_log_prob=False,
            )
            ref_rejected_log_probs = self._get_batch_logps(
                ref_rejected_logits,
                rejected_ids,
                average_log_prob=False,
            )
        
        # Calculate RDPO loss
        pi_logratios = chosen_log_probs - rejected_log_probs
        ref_logratios = ref_chosen_log_probs - ref_rejected_log_probs
        
        logits = pi_logratios - ref_logratios
        
        losses = -F.logsigmoid(self.beta * logits)
        loss = losses.mean()
        
        if return_outputs:
            return loss, outputs
        return loss

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Calculate log probabilities for a batch of sequences."""
        
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (shape {}) and labels (shape {}) have mismatched shapes!".format(
                logits.shape, labels.shape))

        log_probs = F.log_softmax(logits, dim=-1)
        
        # Select the log probabilities of the chosen tokens
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        # Create attention mask to handle padding
        attention_mask = (labels != self.tokenizer.pad_token_id).float()
        token_log_probs = token_log_probs * attention_mask
        
        # Sum up the log probabilities for each sequence
        sequence_log_probs = token_log_probs.sum(dim=-1)
        
        if average_log_prob:
            # Divide by number of non-padding tokens
            sequence_log_probs = sequence_log_probs / attention_mask.sum(dim=-1)
            
        return sequence_log_probs

    def create_optimizer(self):
        """
        Setup the optimizer for model training.
        """
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer_cls = torch.optim.AdamW
        optimizer_kwargs = {
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "lr": self.args.learning_rate,
        }
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return optimizer

    def get_train_dataloader(self):
        """
        Returns the training dataloader.
        """
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        return torch.utils.data.DataLoader(train_dataset, **dataloader_params)

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        loss.backward()
        
        return loss.detach()

    def prediction_step(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on a batch of inputs.
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
                
        return (loss, None, None) 