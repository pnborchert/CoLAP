from ._base import BasePrompt, BaseFewShotPrompt, BaseMultiFewShotPrompt
import torch, json, os
import numpy as np


class PromptDataset(BasePrompt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def post_init(self, **kwargs):
        # label to label_scope id
        self.data["label"] = [self.tokenizer.convert_tokens_to_ids(self.label_mapping[i]) for i in self.data["label"]]
        self.data["label"] = [self.label_scope.tolist().index(i) for i in self.data["label"]]

        # 
    
    def get_prompts(self, data, **kwargs):
        prompts = []
        for i in range(len(data["label"])):
            prompt = self.template.replace("*mask*", self.tokenizer.mask_token)
            prompt = prompt.replace("*sent*", data["sent"][i])
            prompt = prompt.replace("*e1*", data["e1"][i])
            prompt = prompt.replace("*e2*", data["e2"][i])
            prompts.append(prompt)
        self.prompts = prompts

# class FewShotPromptDataset(BaseFewShotPrompt):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def post_init(self, **kwargs):
#         # label to label_scope id
#         self.labels = [self.tokenizer.convert_tokens_to_ids(self.label_mapping[i]) for i in self.labels]
#         self.labels = [self.label_scope.tolist().index(i) for i in self.labels]
    
#     def get_prompts(self, data, **kwargs):
#         prompts = []
#         for i in data:
#             prompt = self.template.replace("*mask*", self.tokenizer.mask_token)
#             prompt = prompt.replace("*sent*", i["sent"])
#             prompt = prompt.replace("*e1*", i["e1"])
#             prompt = prompt.replace("*e2*", i["e2"])
#             prompts.append(prompt)
#         self.prompts = prompts

class MultiFewShotPromptDataset(BaseMultiFewShotPrompt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def post_init(self, **kwargs):
        # label to label_scope id
        self.labels = [self.tokenizer.convert_tokens_to_ids(self.label_mapping[i]) for i in self.labels]
        self.labels = [self.label_scope.tolist().index(i) for i in self.labels]

        self.labels_source = [self.tokenizer.convert_tokens_to_ids(self.label_mapping[i]) for i in self.labels_source]
        self.labels_source = [self.label_scope.tolist().index(i) for i in self.labels_source]
    
    def get_prompts(self, data_source, data_target, **kwargs):
        prompts_source = []
        for i in data_source:
            prompt = self.template.replace("*mask*", self.tokenizer.mask_token)
            prompt = prompt.replace("*sent*", i["sent"])
            prompt = prompt.replace("*e1*", i["e1"])
            prompt = prompt.replace("*e2*", i["e2"])
            prompts_source.append(prompt)
        self.prompts_source = prompts_source

        prompts_target = []
        for i in data_target:
            prompt = self.template.replace("*mask*", self.tokenizer.mask_token)
            prompt = prompt.replace("*sent*", i["sent"])
            prompt = prompt.replace("*e1*", i["e1"])
            prompt = prompt.replace("*e2*", i["e2"])
            prompts_target.append(prompt)
        self.prompts_target = prompts_target

def process_tacred(data):
    out = {
        "sent": [],
        "e1": [],
        "e2": [],
        "label": []
    }
    
    for item in data:
        # Add entity markers <e1>, </e1>, <e2>, </e2>
        tokens = item['token']
        tokens[item['subj_start']] = "<e1> " + tokens[item['subj_start']]
        tokens[item['subj_end']] = tokens[item['subj_end']] + " </e1>"
        tokens[item['obj_start']] = "<e2> " + tokens[item['obj_start']]
        tokens[item['obj_end']] = tokens[item['obj_end']] + " </e2>"
        sent = " ".join(tokens)
        out["sent"].append(sent)
        out["e1"].append(" ".join(item['token'][item['subj_start']:item['subj_end']+1]))
        out["e2"].append(" ".join(item['token'][item['obj_start']:item['obj_end']+1]))
        out["label"].append(item['relation'])

    return out

def process_tacred_episode(data):
    out = {
        "sent": [],
        "e1": [],
        "e2": [],
        "label": []
    }

    if "tokens_translated" in data[0]:
        key_token = "tokens_translated"
        key_entities = "entities_translated"
    else:
        key_token = "tokens"
        key_entities = "entities"

    
    for item in data:
        tokens = item[key_token]
        tokens[item[key_entities][0][0]] = "<e1> " + tokens[item[key_entities][0][0]]
        tokens[item[key_entities][0][1]-1] = tokens[item[key_entities][0][1]-1] + " </e1>"
        tokens[item[key_entities][1][0]] = "<e2> " + tokens[item[key_entities][1][0]]
        tokens[item[key_entities][1][1]-1] = tokens[item[key_entities][1][1]-1] + " </e2>"
        sent = " ".join(tokens)

        out["sent"].append(sent)
        out["e1"].append(" ".join(item[key_token][item[key_entities][0][0]:item[key_entities][0][1]+1]))
        out["e2"].append(" ".join(item[key_token][item[key_entities][1][0]:item[key_entities][1][1]+1]))
        out["label"].append(item['label'])

    return out

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_data(args, lang):

    if args["K"] is not None:
        # load sampled episodes
        train = process_tacred_episode(load_jsonl(os.path.join(args["data_dir"], "tacred", lang, f"episode_{args['seed']}.jsonl")))
        val = process_tacred_episode(load_jsonl(os.path.join(args["data_dir"], "tacred", lang, f"episode_{args['seed']}.jsonl")))
        test = process_tacred_episode(load_jsonl(os.path.join(args["data_dir"], "tacred", lang, f"episode_{args['seed']}.jsonl")))
    else:
        # load full English dataset
        train = process_tacred(json.load(open(os.path.join(args["data_dir"], "tacred", 'train.json'))))
        val = process_tacred(json.load(open(os.path.join(args["data_dir"], "tacred", 'dev.json'))))
        test = process_tacred(json.load(open(os.path.join(args["data_dir"], "tacred", 'test.json'))))

    return train, val, test

def get_episode(data_source, data_target, K, seed):
    episode_source = []
    episode_target = []

    np.random.seed(seed)
    idx = np.random.choice(len(data_target["label"]), K, replace=False)

    for i in idx:
        d_source = {}
        d_target = {}
        for k in data_source.keys():
            d_source[k] = data_source[k][i]
            d_target[k] = data_target[k][i]
        episode_source.append(d_source)
        episode_target.append(d_target)

    episode_source = np.array(episode_source)
    episode_target = np.array(episode_target)

    return {"source": episode_source, "target": episode_target}