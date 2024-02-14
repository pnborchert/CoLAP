import torch, os, glob, safetensors
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput
from transformers import PreTrainedModel

@dataclass
class BaseOutputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Tuple[torch.FloatTensor] = None
    logits_source: Tuple[torch.FloatTensor] = None
    h_source: Tuple[torch.FloatTensor] = None
    h_target: Tuple[torch.FloatTensor] = None

class BaseModelPrompt(PreTrainedModel):
    def __init__(self, encoder, **kwargs):
        super().__init__(encoder.config)

        self.config = encoder.config
        self.encoder = encoder
        self.loss_fct_ce = torch.nn.CrossEntropyLoss()
    
    def load_from_ckpt(self, ckpt_path):
        print("*"*20, f"Loading from {ckpt_path}", "*"*20)
        ckpt_path = glob.glob(os.path.join(ckpt_path, "*", "*.safetensors"))[0]
        safetensors.torch.load_model(self, ckpt_path)
    
    def forward(self, input_ids, attention_mask, mask_pos, label_scope, **kwargs):
        logits = self.encoder(input_ids, attention_mask=attention_mask).logits

        B_logits = logits.size(0)
        i = torch.arange(B_logits).reshape(B_logits, 1, 1)
        j = mask_pos.reshape(B_logits, mask_pos.shape[1], 1)
        k = torch.unique(label_scope)
        logits = logits[i,j,k].squeeze(1)

        out = BaseOutputs(logits = logits)

        return out

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        out = model(**inputs, is_eval= not model.training)
        loss = self.loss_fct_ce(out.logits, labels.view(-1))

        out = BaseOutputs(
            loss=loss,
            logits=out.logits.unsqueeze(0),
        )

        return (loss, out) if return_outputs else loss