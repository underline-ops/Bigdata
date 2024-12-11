import copy
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
from torch.utils.data import Dataset
import transformers
import openmind

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
    "prompt_input": (
        "Premise: {premise}\n"
        "Question: {question}\n"
        "Options:\n"
        "A. {choice1}\n"
        "B. {choice2}\n"
        "C. {choice3}\n"
        "D. {choice4}\n\n"
        "Please select the correct option (A/B/C/D):"
    ),
    "prompt_no_input": (
        "Premise: {premise}\n"
        "Question: {question}\n"
        "Options:\n"
        "A. {choice1}\n"
        "B. {choice2}\n\n"
        "Please select the correct option (A/B):"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(openmind.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    num_choices: int = field(
        default=2,
        metadata={"help": "Number of choices per question."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        with open(f, mode=mode) as f:
            jdict = json.load(f)
    else:
        jdict = json.load(f)
    return jdict

def preprocess(
    sources: Sequence[str],
    labels: Sequence[int],
    tokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    tokenized = tokenizer(
        sources,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return dict(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        labels=torch.tensor(labels),
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer, num_choices: int):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Pairing mirrored and non-mirrored examples...")
        # 创建一个字典，以非镜像样本的id作为键，镜像样本作为值
        paired_data = {}
        for example in list_data_dict:
            base_id = example["id"] if not example["mirrored"] else example["id"] - 1000
            if base_id not in paired_data:
                paired_data[base_id] = {}
            if example["mirrored"]:
                paired_data[base_id]["mirrored"] = example
            else:
                paired_data[base_id]["original"] = example

        logging.warning("Formatting inputs...")
        sources = []
        labels = []
        for base_id, pair in paired_data.items():
            original = pair.get("original")
            mirrored = pair.get("mirrored")

            if original and original.get("correct_answer"):
                premise = original["premise"]
                question = original["question"]
                choice1 = original["choice1"]
                choice2 = original["choice2"]
                # 处理多选情况
                choice3 = original.get("choice3", "")
                choice4 = original.get("choice4", "")
                if num_choices > 2 and choice3 and choice4:
                    prompt = PROMPT_DICT["prompt_input"].format(
                        premise=premise,
                        question=question,
                        choice1=choice1,
                        choice2=choice2,
                        choice3=choice3,
                        choice4=choice4
                    )
                else:
                    prompt = PROMPT_DICT["prompt_no_input"].format(
                        premise=premise,
                        question=question,
                        choice1=choice1,
                        choice2=choice2
                    )
                sources.append(prompt)
                # 将正确答案转换为标签
                answer = original.get("correct_answer", "").strip().upper()
                label = self.convert_answer_to_label(answer)
                if label is not None:
                    labels.append(label)

            if mirrored and mirrored.get("correct_answer"):
                premise = mirrored["premise"]
                question = mirrored["question"]
                choice1 = mirrored["choice1"]
                choice2 = mirrored["choice2"]
                # 处理多选情况
                choice3 = mirrored.get("choice3", "")
                choice4 = mirrored.get("choice4", "")
                if num_choices > 2 and choice3 and choice4:
                    prompt = PROMPT_DICT["prompt_input"].format(
                        premise=premise,
                        question=question,
                        choice1=choice1,
                        choice2=choice2,
                        choice3=choice3,
                        choice4=choice4
                    )
                else:
                    prompt = PROMPT_DICT["prompt_no_input"].format(
                        premise=premise,
                        question=question,
                        choice1=choice1,
                        choice2=choice2
                    )
                sources.append(prompt)
                # 将正确答案转换为标签
                answer = mirrored.get("correct_answer", "").strip().upper()
                label = self.convert_answer_to_label(answer)
                if label is not None:
                    labels.append(label)

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, labels, tokenizer)

        try:
            self.input_ids = data_dict["input_ids"]
            self.attention_mask = data_dict["attention_mask"]
            self.labels = data_dict["labels"]
        except KeyError as e:
            raise KeyError("Required keys are missing in data_dict") from e

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            attention_mask=self.attention_mask[i],
            labels=self.labels[i],
        )

    @staticmethod
    def convert_answer_to_label(answer: str) -> Optional[int]:
        """Convert answer string to label index."""
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        if answer in mapping:
            return mapping[answer]
        else:
            # 尝试从选项文本推断标签
            reverse_mapping = {
                "The grass was cut.": 0,
                "The sun was rising.": 1,
                "The woman knew her friend was going through a hard time.": 0,
                "The woman felt that her friend took advantage of her kindness.": 1,
                # 根据实际数据添加更多映射
            }
            return reverse_mapping.get(answer, None)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: object

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([instance["input_ids"] for instance in instances])
        attention_mask = torch.stack([instance["attention_mask"] for instance in instances])
        labels = torch.stack([instance["labels"] for instance in instances])
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

def make_supervised_data_module(tokenizer, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        num_choices=training_args.num_choices
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = openmind.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )

    tokenizer = openmind.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    trainer = openmind.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"]
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    train()
