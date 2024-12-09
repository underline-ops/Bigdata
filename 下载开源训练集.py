#先安装datasets库，以便连接Hugging Face Datasets Hub
#pip install datasets

import os
import json
from datasets import load_dataset, DatasetDict


def save_split_as_json(split, split_name, output_dir, field_mapping=None):
    """
    将数据集拆分保存为 JSON 文件，并可进行字段映射。

    Args:
        split (Dataset): 数据集的拆分（train/validation/test）。
        split_name (str): 拆分名称。
        output_dir (str): 输出目录。
        field_mapping (dict, optional): 字段映射字典。例如：
            {
                "premise": "context",
                "question": "question_text",
                "choice1": "optionA",
                "choice2": "optionB",
                "correct_answer": "label"
            }
    """
    output_path = os.path.join(output_dir, f"{split_name}.json")
    print(f"Saving {split_name} split to {output_path}...")

    # 转换为列表字典
    data = split.to_dict()
    records = [dict(zip(data.keys(), values)) for values in zip(*data.values())]

    if field_mapping:
        mapped_records = []
        for record in records:
            mapped_record = {}
            for new_key, old_key in field_mapping.items():
                mapped_record[new_key] = record.get(old_key, "")
            mapped_records.append(mapped_record)
        records = mapped_records

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
    print(f"Saved {split_name} split successfully.")


def download_and_save(dataset_name, config_name=None, output_dir="downloaded_datasets", field_mapping=None):
    """
    下载指定的数据集并保存为 JSON 格式，可选字段映射。

    Args:
        dataset_name (str): 数据集名称。
        config_name (str, optional): 数据集配置名称。默认为 None。
        output_dir (str, optional): 输出目录。默认为 "downloaded_datasets"。
        field_mapping (dict, optional): 字段映射字典。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    print(f"Loading dataset '{dataset_name}'" + (f" with config '{config_name}'" if config_name else "") + "...")
    if config_name:
        dataset = load_dataset(dataset_name, config_name)
    else:
        dataset = load_dataset(dataset_name)
    print("Dataset loaded successfully.")

    # 如果是 DatasetDict，则有多个拆分
    if isinstance(dataset, DatasetDict):
        for split_name in dataset.keys():
            split = dataset[split_name]
            save_split_as_json(split, split_name, output_dir, field_mapping)
    else:
        # 单一拆分的数据集
        save_split_as_json(dataset, "data", output_dir, field_mapping)


def main():
    # 在这里设置参数
    dataset_name = "copa"  # Hugging face 的训练集名称
    config_name = None  # 如果有配置名称，则设置，否则为 None
    output_dir = "downloaded_datasets/copa_data"  # 设置输出目录，注意按照内容放到不同目录内
    field_mapping = {
        "premise": "premise",
        "question": "question",
        "choice1": "choice1",
        "choice2": "choice2",
        "correct_answer": "label"  # 根据实际数据集字段调整
    }

    # 调用下载和保存函数
    download_and_save(dataset_name, config_name, output_dir, field_mapping)


if __name__ == "__main__":
    main()
