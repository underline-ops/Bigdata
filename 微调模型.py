import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score


def load_json_dataset(file_path):
    """
    从 JSON 文件加载数据集。

    Args:
        file_path (str): JSON 文件路径。

    Returns:
        Dataset: Hugging Face Dataset 对象。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)


def prepare_datasets(train_path, val_path=None, test_path=None):
    """
    准备训练、验证和测试数据集。

    Args:
        train_path (str): 训练集 JSON 文件路径。
        val_path (str, optional): 验证集 JSON 文件路径。如果未提供，则从训练集中划分。
        test_path (str, optional): 测试集 JSON 文件路径。

    Returns:
        DatasetDict: 包含训练、验证和测试数据集的字典。
    """
    train_dataset = load_json_dataset(train_path)

    if val_path and os.path.exists(val_path):
        val_dataset = load_json_dataset(val_path)
    else:
        # 从训练集中划分 10% 作为验证集
        train_val = train_dataset.train_test_split(test_size=0.1)
        train_dataset = train_val['train']
        val_dataset = train_val['test']

    if test_path and os.path.exists(test_path):
        test_dataset = load_json_dataset(test_path)
    else:
        test_dataset = None

    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })


def tokenize_function(examples, tokenizer):
    """
    对数据集进行分词。

    Args:
        examples (dict): 数据样本。
        tokenizer (PreTrainedTokenizer): Hugging Face 分词器。

    Returns:
        dict: 分词后的输入。
    """
    return tokenizer(
        examples['premise'],
        examples['question'],
        truncation=True,
        padding='max_length',
        max_length=128
    )


def compute_metrics(pred):
    """
    计算评估指标。

    Args:
        pred: Trainer 预测结果。

    Returns:
        dict: 评估指标。
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
    }


def main():
    """
    主函数：设置参数并执行微调流程。
    """
    # 设置参数
    datasets_to_download = [
        {
            "dataset_name": "ReClor",
            "train_path": "datasets/ReClor/reclor_train.json",
            "val_path": "datasets/ReClor/reclor_validation.json",
            "test_path": "datasets/ReClor/reclor_test.json"
        },
        {
            "dataset_name": "LogiQA",
            "train_path": "datasets/LogiQA/logiqa_train.json",
            "val_path": "datasets/LogiQA/logiqa_validation.json",
            "test_path": "datasets/LogiQA/logiqa_test.json"
        }
    ]

    model_name = "path/to/Qwen1.5"  # 替换为 Qwen1.5 模型的 Hugging Face 名称或本地路径
    output_base_dir = "fine_tuned_models"  # 模型保存的基础目录

    # 检查 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    for ds in datasets_to_download:
        dataset_name = ds["dataset_name"]
        train_path = ds["train_path"]
        val_path = ds["val_path"]
        test_path = ds["test_path"]

        print(f"\n开始处理数据集: {dataset_name}")

        # 准备数据集
        datasets = prepare_datasets(train_path, val_path, test_path)

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 分词
        print("分词数据集...")
        tokenized_datasets = datasets.map(lambda x: tokenize_function(x, tokenizer), batched=True)

        # 加载模型
        print("加载预训练模型...")
        num_labels = len(set(datasets['train']['correct_answer']))  # 根据任务调整
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        model.to(device)

        # 设置训练参数
        output_dir = os.path.join(output_base_dir, f"{dataset_name}_Qwen1.5")
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,  # 根据 GPU 显存调整
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=True,  # 启用混合精度训练
            dataloader_num_workers=4,
        )

        # 初始化 Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # 训练模型
        print("开始训练...")
        trainer.train()

        # 保存模型
        print(f"保存模型到 {output_dir}...")
        trainer.save_model()

        # 评估模型
        print("评估模型...")
        results = trainer.evaluate()
        print(f"评估结果: {results}")

        # 预测测试集（如果有）
        if 'test' in datasets and datasets['test']:
            print("进行测试集预测...")
            predictions = trainer.predict(tokenized_datasets['test'])
            preds = predictions.predictions.argmax(-1)
            test_dataset = datasets['test'].to_pandas()
            test_dataset['predictions'] = preds
            test_output_path = os.path.join(output_dir, f"{dataset_name}_test_predictions.json")
            test_dataset.to_json(test_output_path, orient='records', force_ascii=False, indent=4)
            print(f"测试集预测结果已保存至 {test_output_path}")


if __name__ == "__main__":
    main()
