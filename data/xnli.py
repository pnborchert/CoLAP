from datasets import load_dataset
import numpy as np
from ._base import BasePrompt, BaseFewShotPrompt, BaseMultiFewShotPrompt

class PromptDataset(BasePrompt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_prompts(self, data, **kwargs):
        prompts = []
        for i in range(len(data["label"])):
            prompt = self.template.replace("*mask*", self.tokenizer.mask_token)
            prompt = prompt.replace("*premise*", data["premise"][i])
            prompt = prompt.replace("*hypothesis*", data["hypothesis"][i])
            prompts.append(prompt)
        self.prompts = prompts

# class FewShotPromptDataset(BaseFewShotPrompt):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def get_prompts(self, data, **kwargs):
#         prompts = []
#         for i in data:
#             prompt = self.template.replace("*mask*", self.tokenizer.mask_token)
#             prompt = prompt.replace("*premise*", i["premise"])
#             prompt = prompt.replace("*hypothesis*", i["hypothesis"])
#             prompts.append(prompt)
#         self.prompts = prompts

class MultiFewShotPromptDataset(BaseMultiFewShotPrompt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_prompts(self, data_source, data_target, **kwargs):
        prompts_source = []
        for i in data_source:
            prompt = self.template.replace("*mask*", self.tokenizer.mask_token)
            prompt = prompt.replace("*premise*", i["premise"])
            prompt = prompt.replace("*hypothesis*", i["hypothesis"])
            prompts_source.append(prompt)
        self.prompts_source = prompts_source

        prompts_target = []
        for i in data_target:
            prompt = self.template.replace("*mask*", self.tokenizer.mask_token)
            prompt = prompt.replace("*premise*", i["premise"])
            prompt = prompt.replace("*hypothesis*", i["hypothesis"])
            prompts_target.append(prompt)
        self.prompts_target = prompts_target

def load_data(args, lang):

    dataset = load_dataset("xnli", lang)
    train = dataset["train"].to_dict()
    val = dataset["validation"].to_dict()
    test = dataset["test"].to_dict()

    return train, val, test

def dict_per_label(data):
    x_out = {str(k):[] for k in np.unique(data["label"])}
    for i in range(len(data["label"])):
        x_out[str(data["label"][i])].append({"premise": data["premise"][i], "hypothesis": data["hypothesis"][i], "label": data["label"][i]})
    return x_out

def get_episode(data_source, data_target, K, seed):
    np.random.seed(seed)
    num_labels = len(np.unique(data_target["label"]))
    K_per_label = K // num_labels

    data_source = dict_per_label(data_source)
    data_source = {k:np.array(v) for k, v in data_source.items()}

    data_target = dict_per_label(data_target)
    data_target = {k:np.array(v) for k, v in data_target.items()}

    episode_source = []
    episode_target = []
    for k, v in data_target.items():
        idx = np.random.choice(min(len(v), len(data_source[k])), K_per_label, replace=False)
        episode_target += list(v[idx])
        episode_source += list(data_source[k][idx])
    
    # assign remaining samples
    for _ in range(K % num_labels):
        k = np.random.choice(list(data_target.keys()))
        v = data_target[k]
        idx = np.random.choice(min(len(v), len(data_source[k])))
        episode_target += [v[idx]]
        episode_source += [data_source[k][idx]]

    episode_source = np.array(episode_source)
    episode_target = np.array(episode_target)
    return {"source": episode_source, "target": episode_target}