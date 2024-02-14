import torch, os, glob, safetensors
import torch.nn.functional as F
from transformers import PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput

@dataclass
class CoLAPOutputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Tuple[torch.FloatTensor] = None
    logits_source: Tuple[torch.FloatTensor] = None
    h_source: Tuple[torch.FloatTensor] = None
    h_target: Tuple[torch.FloatTensor] = None

class CoLAP(PreTrainedModel):
    """Contrastive Language Alignment with Prompting"""

    def __init__(self, encoder, args):
        super().__init__(encoder.config)

        self.config = encoder.config
        self.encoder = encoder
        self.add_loss_ce = True
        self.add_loss_xrcl = args["xrcl"]
        self.add_loss_xccl = args["xccl"]
        self.layer = 9 # layer to extract hidden states (0-11 for xlmr-base)

        self.loss_fct_ce = torch.nn.CrossEntropyLoss()
    
    def load_from_ckpt(self, ckpt_path):
        print("*"*20, f"Loading from {ckpt_path}", "*"*20)
        ckpt_path = glob.glob(os.path.join(ckpt_path, "*", "*.safetensors"))[0]
        safetensors.torch.load_model(self, ckpt_path)
    
    def loss_fct_xrcl(self, h_s, h_t, temp=0.05, eps=1e-8):
        """
        Cross-lingual Representation Contrastive Loss

        h_s: hidden state (mask representation) from source language
        h_t: hidden state (mask representation) from target language
        temp: temperature

        """

        s = F.normalize(h_s, dim=1)
        t = F.normalize(h_t, dim=1)
        sim = t @ s.transpose(0,1)
        sim = torch.exp(sim / temp)
        pos = (sim.diag()).sum(-1) + eps
        neg = sim.sum(-1) - sim.diag() + eps
        loss = -torch.log(pos / (pos + neg)).sum()
        return loss / (h_s.size(0))
    
    def loss_fct_xccl(self, h_s, h_t, l_s, l_t, temp=0.05, eps=1e-8):
        """
        Cross-lingual Class Contrastive Loss

        h_s: hidden state (mask representation) from source language
        h_t: hidden state (mask representation) from target language
        l_s: label from source language
        l_t: label from target language
        temp: temperature

        """

        s = F.normalize(h_s, dim=1)
        t = F.normalize(h_t, dim=1)
        sim = t @ s.transpose(0,1)
        sim = torch.exp(sim / temp)
        mask = (l_s.unsqueeze(0) == l_t.unsqueeze(1)).float().to(h_s.device)
        num = (sim * mask).sum(-1) + eps
        denom = sim.sum(-1) + eps
        loss = -torch.log(num / denom).sum()
        return loss / (l_t.size(0))
    
    def forward(self, *args, **kwargs):
        if kwargs.get("is_eval", False):
            return self.forward_monolingual(*args, **kwargs)
        else:
            return self.forward_multilingual(*args, **kwargs)

    def forward_monolingual(self, input_ids, attention_mask, mask_pos, label_scope, **kwargs):
        logits = self.encoder(input_ids, attention_mask=attention_mask).logits

        B_logits = logits.size(0)
        i = torch.arange(B_logits).reshape(B_logits, 1, 1)
        j = mask_pos.reshape(B_logits, mask_pos.shape[1], 1)
        k = torch.unique(label_scope)
        logits = logits[i,j,k].squeeze(1) 

        out = CoLAPOutputs(
            logits = logits,
            )

        return out

    def forward_multilingual(self, input_ids_source, attention_mask_source, mask_pos_source, input_ids_target, attention_mask_target, mask_pos_target, label_scope, **kwargs):

        out_source = self.encoder(input_ids_source, attention_mask=attention_mask_source)
        out_target = self.encoder(input_ids_target, attention_mask=attention_mask_target)

        logits_source = out_source.logits
        logits_target = out_target.logits

        B_logits = logits_source.size(0)
        i = torch.arange(B_logits).reshape(B_logits, 1, 1)
        j = mask_pos_source.reshape(B_logits, mask_pos_source.shape[1], 1)
        k = torch.unique(label_scope)
        logits_source = logits_source[i,j,k].squeeze(1)
        h_source = out_source.hidden_states[self.layer][torch.arange(B_logits).unsqueeze(1), mask_pos_source, :].squeeze(1)

        B_logits = logits_target.size(0)
        i = torch.arange(B_logits).reshape(B_logits, 1, 1)
        j = mask_pos_target.reshape(B_logits, mask_pos_target.shape[1], 1)
        k = torch.unique(label_scope)
        logits_target = logits_target[i,j,k].squeeze(1)
        h_target = out_target.hidden_states[self.layer][torch.arange(B_logits).unsqueeze(1), mask_pos_target, :].squeeze(1)

        out = CoLAPOutputs(
            logits_source = logits_source,
            logits = logits_target,
            h_source = h_source,
            h_target = h_target,
            )
        
        return out
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels_source = inputs.pop("labels_source", None)
        out = model(**inputs, is_eval= not model.training)

        if self.add_loss_ce:
            loss_ce = self.loss_fct_ce(out.logits, labels.view(-1))
        else:
            loss_ce = torch.tensor(0.0)
        
        if (self.add_loss_xrcl) and (model.training):
            loss_xrcl = self.loss_fct_xrcl(h_s=out.h_source, h_t=out.h_target)
        else:
            loss_xrcl = torch.tensor(0.0)
        
        if (self.add_loss_xccl) and (model.training):
            loss_xccl = self.loss_fct_xccl(h_s=out.h_source, h_t=out.h_target, l_s=labels_source.view(-1), l_t=labels.view(-1))
        else:
            loss_xccl = torch.tensor(0.0)

        loss = loss_ce + loss_xrcl + loss_xccl

        out = CoLAPOutputs(
            loss=loss,
            logits=out.logits.unsqueeze(0),
        )

        return (loss, out) if return_outputs else loss