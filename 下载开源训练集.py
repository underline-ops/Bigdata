#先安装datasets库，以便连接Hugging Face Datasets Hub。
#pip install datasets

import os
import json
from datasets import load_dataset


def main():
    # 加载 "pkavumba/balanced-copa" 数据集
    ds = load_dataset("pkavumba/balanced-copa")

    # 创建保存目录 "训练集"（如果不存在）
    output_dir = "训练集"
    os.makedirs(output_dir, exist_ok=True)

    # 选择训练集
    train_split = ds['train']

    # 处理每条记录，将 'label' 转换为 'correct_answer'
    processed_records = []
    for record in train_split:
        label = record.get("label")
        correct_answer = ""

        # 根据 'label' 的值映射到 'choice1' 或 'choice2'
        if label == 1:
            correct_answer = record.get("choice1", "")
        elif label == 2:
            correct_answer = record.get("choice2", "")
        # 如果有更多的选项，可以在这里继续添加 elif 语句

        # 构建新的记录字典
        processed_record = {
            "id": record.get("id"),
            "premise": record.get("premise", ""),
            "question": record.get("question", ""),
            "choice1": record.get("choice1", ""),
            "choice2": record.get("choice2", ""),
            "correct_answer": correct_answer,
            "mirrored": record.get("mirrored", False)
        }
        processed_records.append(processed_record)

    # 修改保存后的文件名
    output_path = os.path.join(output_dir, "train.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_records, f, ensure_ascii=False, indent=4)

    print(f"训练集已成功保存至 {output_path}，")


if __name__ == "__main__":
    main()
