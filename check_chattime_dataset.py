from datasets import load_dataset
import pandas as pd
import re
import numpy as np

dataset_path = "ChengsenWang/ChatTime-1-Pretrain-1M"

dataset = load_dataset(
    "csv",
    data_files=f"https://huggingface.co/datasets/{dataset_path}/resolve/main/ChatTime-1-Pretrain-1M.csv",
    split="train",
)

print(dataset)
print("Column names:", dataset.column_names)
print("Number of rows:", len(dataset))

print("\n===== First example =====")
print(dataset[0])

print("\n===== First text head =====")
print(dataset[0]["text"][:2000])

print("\n===== First 5 examples short =====")
for i in range(5):
    text = dataset[i]["text"]
    print(f"\n--- example {i} ---")
    print("length chars:", len(text))
    print("num whitespace tokens:", len(text.split()))
    print(text[:500])