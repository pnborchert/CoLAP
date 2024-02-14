import torch
import numpy as np

# quickstart example
def to_colap_input(data_target, label, tokenizer, data_source=None, label_source=None):

    inputs_target = tokenizer(data_target, return_tensors="pt", padding=True, truncation=True)
    mask_pos_target = torch.where(inputs_target["input_ids"] == tokenizer.mask_token_id)[1].reshape(-1, 1)

    label_enc_target = tokenizer(label, add_special_tokens=False, return_tensors="pt")["input_ids"]
    label_scope = torch.unique(label_enc_target)
    labels_target = torch.tensor([label_scope.tolist().index(l) for l in label_enc_target.reshape(-1)])

    if data_source is not None:
        inputs_source = tokenizer(data_source, return_tensors="pt", padding=True, truncation=True)
        mask_pos_source = torch.where(inputs_source["input_ids"] == tokenizer.mask_token_id)[1].reshape(-1, 1)

        label_enc_source = tokenizer(label_source, add_special_tokens=False, return_tensors="pt")["input_ids"]
        labels_source = torch.tensor([label_scope.tolist().index(l) for l in label_enc_source.reshape(-1)])
    
        inputs = {
            "input_ids_target": inputs_target["input_ids"],
            "attention_mask_target": inputs_target["attention_mask"],
            "mask_pos_target": mask_pos_target,
            "labels": labels_target,
            "label_scope": label_scope,
            "input_ids_source": inputs_source["input_ids"],
            "attention_mask_source": inputs_source["attention_mask"],
            "mask_pos_source": mask_pos_source,
            "labels_source": labels_source,
        }
    else:
        inputs = {
            "input_ids": inputs_target["input_ids"],
            "attention_mask": inputs_target["attention_mask"],
            "mask_pos": mask_pos_target,
            "label_scope": label_scope,
            "is_eval": True,
        }
    
    return inputs
    
class BasePrompt(torch.utils.data.Dataset):

    def __init__(self, data, tokenizer, args, **kwargs):

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
        self.template = args["template"]
        self.label_mapping = args["label_mapping"]
        self.task = args["task"]
        self.lang = args["target_lang"]
        self.label_scope = torch.unique(torch.as_tensor(self.tokenizer.convert_tokens_to_ids(self.label_mapping.values())))
        self.tokenizer.truncation_side = "right" if self.lang in ["ar", "ur"] else "left"

        # fill prompt templates
        self.get_prompts(data=self.data)
        self.prompts = np.array(self.prompts)

        # preprocessing post init
        self.post_init(**kwargs)
    
    def post_init(self, **kwargs):
        pass

    def get_prompts(self, **kwargs):
        raise NotImplementedError 

    def __len__(self):
        return len(self.data["label"])
    
    def __getitem__(self, idx):

        prompt = self.prompts[idx]
        label = self.data["label"][idx]
        
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        mask_pos = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

        # if len(self) == 22631:
        #     print(prompt, label)

        if mask_pos.shape[0] == 0:
            print(prompt, label)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "mask_pos": mask_pos,
            "labels": torch.tensor(label),
            "label_scope": self.label_scope,
            }

class BaseFewShotPrompt(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, args, **kwargs):

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
        self.template = args["template"]
        self.label_mapping = args["label_mapping"]
        self.lang = args["target_lang"]
        self.task = args["task"]
        self.label_scope = torch.unique(torch.as_tensor(self.tokenizer.convert_tokens_to_ids(self.label_mapping.values())))
        self.tokenizer.truncation_side = "right" if self.lang in ["ar", "ur"] else "left"

        # fill prompt templates
        self.get_prompts(data=self.data)
        self.labels = [i["label"] for i in self.data]
        self.prompts = np.array(self.prompts)

        # preprocessing post init
        self.post_init(**kwargs)
    
    def post_init(self, **kwargs):
        pass

    def get_prompts(self, **kwargs):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
            
        prompt = self.prompts[idx]
        label = self.labels[idx]
        
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        mask_pos = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "mask_pos": mask_pos,
            "labels": torch.tensor(label),
            "label_scope": self.label_scope,
            }
    
class BaseMultiFewShotPrompt(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, args, **kwargs):

        self.data_source = data["source"]
        self.data_target = data["target"]
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
        self.template = args["template"]
        self.label_mapping = args["label_mapping"]
        self.source_lang = args["source_lang"]
        self.target_lang = args["target_lang"]
        self.task = args["task"]
        self.label_scope = torch.unique(torch.as_tensor(self.tokenizer.convert_tokens_to_ids(self.label_mapping.values())))
        self.trunc_source = "right" if self.source_lang in ["ar", "ur"] else "left"
        self.trunc_target = "right" if self.target_lang in ["ar", "ur"] else "left"
        
        # fill prompt templates
        self.get_prompts(data_source=self.data_source, data_target=self.data_target)
        self.labels = [i["label"] for i in self.data_target]
        self.labels_source = [i["label"] for i in self.data_source]
        self.prompts_source = np.array(self.prompts_source)
        self.prompts_target = np.array(self.prompts_target)

        # preprocessing post init
        self.post_init(**kwargs)
    
    def post_init(self, **kwargs):
        pass

    def get_prompts(self, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
                
            prompt_source = self.prompts_source[idx]
            prompt_target = self.prompts_target[idx]
            label_source = self.labels_source[idx]
            label = self.labels[idx]

            self.tokenizer.truncation_side = self.trunc_source
            inputs_source = self.tokenizer(
                prompt_source,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

            self.tokenizer.truncation_side = self.trunc_target
            inputs_target = self.tokenizer(
                prompt_target,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
    
            mask_pos_source = torch.where(inputs_source["input_ids"] == self.tokenizer.mask_token_id)[1]
            mask_pos_target = torch.where(inputs_target["input_ids"] == self.tokenizer.mask_token_id)[1]
    
            return {
                "input_ids_source": inputs_source["input_ids"].squeeze(0),
                "attention_mask_source": inputs_source["attention_mask"].squeeze(0),
                "mask_pos_source": mask_pos_source,
                "input_ids_target": inputs_target["input_ids"].squeeze(0),
                "attention_mask_target": inputs_target["attention_mask"].squeeze(0),
                "mask_pos_target": mask_pos_target,
                "labels": torch.tensor(label),
                "labels_source": torch.tensor(label_source),
                "label_scope": self.label_scope,
                }
