import argparse, json, os
import numpy as np
from transformers import AutoTokenizer, XLMRobertaTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
from transformers.utils import logging
from sklearn.metrics import f1_score, accuracy_score

import models

parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str, required=True)
parser.add_argument('--target_lang', type=str, required=True)
parser.add_argument("--plm", type=str, required=True)
parser.add_argument('--lr', type=float, default=2e-5) 
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=5) # 10 for few-shot
parser.add_argument('--eval_steps', type=int, default=300)
parser.add_argument("--cased", action="store_true", default=False)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--output_dir", type=str, default="checkpoints")
parser.add_argument('--data_dir', type=str, default="datasets")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--fp16', dest="fp16", action='store_true')

# few-shot settings
parser.add_argument('--K', type=int)
parser.add_argument('--source_lang', type=str)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model', type=str, default="colap")
parser.add_argument("--load_ckpt", type=str, default=None)
parser.add_argument("--eval_zs", action="store_true", default=False)
parser.add_argument("--eval_fs", action="store_true", default=False)

# CoLAP settings
parser.add_argument('--xrcl', action='store_true')
parser.add_argument('--no-xrcl', dest='xrcl', action='store_false')
parser.set_defaults(xrcl=True)
parser.add_argument('--xccl', action='store_true')
parser.add_argument('--no-xccl', dest='xccl', action='store_false')
parser.set_defaults(xccl=False)

args = vars(parser.parse_args())

def load_mapping(path):
    return json.load(open(path))

def load_template(path):
    with open(path, encoding="utf-8") as fp:
        lines = fp.readlines()
    lines = [l.replace("\n", "") for l in lines]
    return lines[0]

def load_model(args):

    encoder = AutoModelForMaskedLM.from_pretrained(args["plm"], output_hidden_states=True)

    if args["model"] == "colap":
        if args["K"] is not None:
            model = models.CoLAP(encoder, args)
        else:
            model = models.BaseModelPrompt(encoder)
    else:
        raise NotImplementedError(f"Model {args['model']} not implemented")
    
    return model

def get_dataset(tokenizer, args):

    if args["task"] == "xnli":
        from data.xnli import load_data, get_episode, PromptDataset, MultiFewShotPromptDataset
    elif args["task"] == "amnli":
        from data.amnli import load_data, get_episode, PromptDataset, MultiFewShotPromptDataset
    elif args["task"] == "tacred":
        from data.tacred import load_data, get_episode, PromptDataset, MultiFewShotPromptDataset

    is_few_shot = args["K"] is not None

    if is_few_shot:
        if args["model"] == "colap":
            assert args["source_lang"] is not None, "Provide '--source_lang'"
            assert any([args["xrcl"], args["xccl"]]), "At least one contrastive loss should be enabled: '--xrcl' | '--xccl'"
        
        train_source, _, _ = load_data(args, lang=args["source_lang"])
        train_target, val_target, test_target = load_data(args, lang=args["target_lang"])
        train_source_target = get_episode(train_source, train_target, args["K"], args["seed"])

        ds_train = MultiFewShotPromptDataset(data=train_source_target, tokenizer=tokenizer, args=args)
        ds_valid = PromptDataset(data=val_target, tokenizer=tokenizer, args=args)
        ds_test = PromptDataset(data=test_target, tokenizer=tokenizer, args=args)

        return ds_train, ds_valid, ds_test

    else:
        # fine-tuning on task data
        if (args["task"] == "amnli") and (args["target_lang"] == "en"):
            raise Warning("AmNLI does not include English data. Use XNLI for fine-tuning in English instead.")

        train, val, test = load_data(args, lang=args["target_lang"])
        ds_train = PromptDataset(data=train, tokenizer=tokenizer, args=args)
        ds_valid = PromptDataset(data=val, tokenizer=tokenizer, args=args)
        ds_test = PromptDataset(data=test, tokenizer=tokenizer, args=args)

        return ds_train, ds_valid, ds_test

# compute metrics
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    labels = labels.reshape(-1)
    preds = np.argmax(logits, axis=-1).reshape(-1)
    preds = preds[:len(labels)] # remove padding
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

def evaluate(trainer, ds, args, prefix=None):
    out = trainer.predict(ds)
    metrics = dict(out.metrics)
    
    # write to file
    file_name = "results.json" if (prefix is None) else f"{prefix}-results.json"
    with open(os.path.join(args["output_dir"], args["run_name"], file_name), "w") as fp:
        json.dump(metrics, fp)
    
    return metrics

class PromptTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model.compute_loss(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
            )

def main(args):
    logger = logging.get_logger("transformers")

    is_few_shot = args["K"] is not None

    args["run_name"] = "-".join([args["task"], args["plm"].replace("/", "-"), args["target_lang"], args["model"]])

    if is_few_shot:
        args["run_name"] += f"-{args['source_lang']}"
        args["run_name"] += f"-{args['K']}"
        args["run_name"] += f"-{args['seed']}"
        args["run_name"] += f"-t" if (args["load_ckpt"] is None) else f"-st"

        if args["xrcl"]:
            args["run_name"] += "-xrcl"
        if args["xccl"]:
            args["run_name"] += "-xccl"
    
    # load tokenizer
    if args["plm"] in ["xlm-roberta-base", "microsoft/infoxlm-base"]:
        tokenizer = XLMRobertaTokenizer.from_pretrained(args["plm"], do_lower_case=not args["cased"])
        tokenizer.do_lower_case = not args["cased"] # https://github.com/huggingface/transformers/issues/9122
    else:
        tokenizer = AutoTokenizer.from_pretrained(args["plm"], do_lower_case=not args["cased"])
    
    # load model
    model = load_model(args)

    if args["task"] == "tacred":
        # Extend vocab with marker tokens: <e1>, </e1>, <e2>, </e2>
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
        model.encoder.resize_token_embeddings(len(tokenizer))
        model.encoder.config.vocab_size = len(tokenizer)
    
    # load data
    args["template"] = load_template(os.path.join(args["data_dir"], args['task'], "template.txt"))
    args["label_mapping"] = load_mapping(os.path.join(args["data_dir"], args['task'], "mapping.json"))
    args["num_labels"] = len(args["label_mapping"])
    ds_train, ds_valid, ds_test = get_dataset(tokenizer, args)

    args["batch_size"] = min(args["batch_size"], len(ds_train))
    args["max_steps"] = args["epochs"] * len(ds_train) // args["batch_size"] // args["gradient_accumulation_steps"]

    print("Length of train dataset:", len(ds_train))
    print("Length of valid dataset:", len(ds_valid))
    print("Length of test dataset:", len(ds_test))

    # setup trainer
    training_args = TrainingArguments(
        output_dir=os.path.join(args["output_dir"], args["run_name"]),
        do_train=True,
        do_eval=True,
        do_predict=True,
        remove_unused_columns=False,
        evaluation_strategy="steps", 
        eval_steps = args["eval_steps"] ,
        per_device_train_batch_size=args["batch_size"],
        per_device_eval_batch_size=args["eval_batch_size"],
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        learning_rate=args["lr"],
        max_steps = args["max_steps"],
        lr_scheduler_type="linear",
        save_strategy="no",
        save_total_limit = 1,
        label_names=["labels"],
        run_name=args["run_name"],
        fp16=args["fp16"],
        use_cpu = args["device"] == "cpu",
    )

    if not is_few_shot:
        training_args.save_strategy = "steps"
        training_args.save_steps = args["eval_steps"]
        training_args.save_total_limit = 1
        training_args.load_best_model_at_end = True
        training_args.save_only_model = True
    else:
        training_args.eval_steps = 99999 # no evaluation during few-shot training

    trainer = PromptTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        compute_metrics=compute_metrics,
    )

    # load ckpt
    if args["load_ckpt"] is not None:
        trainer.model.load_from_ckpt(args["load_ckpt"])
    
    if is_few_shot:
        if args["eval_zs"]:
            # zero-shot evaluation
            res = evaluate(trainer=trainer, ds=ds_test, args=args, prefix="zs")
            logger.warning(f"Zero-Shot: {res}")
        
        if args["eval_fs"]:
            # few-shot training
            trainer.train()
            res = evaluate(trainer=trainer, ds=ds_test, args=args, prefix="fs")
            logger.warning(f"Few-Shot: {res}")
    else:
        # fine-tuning on task data
        trainer.train()
        res = evaluate(trainer=trainer, ds=ds_test, args=args)
        logger.warning(f"Test: {res}")

if __name__ == "__main__":
    main(args)