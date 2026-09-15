from datasets import load_dataset


SFT_URL = (
    "https://huggingface.co/datasets/"
    "ChengsenWang/ChatTime-1-Finetune-100K/"
    "resolve/main/ChatTime-1-Finetune-100K.csv"
)

sft = load_dataset(
    "csv",
    data_files=SFT_URL,
    split="train",
)


def is_tsqa_text(text):
    text = str(text).lower()

    has_all_choices = (
        "(a)" in text
        and "(b)" in text
        and "(c)" in text
    )

    has_analysis_instruction = (
        "answer the following question" in text
        or "analyzing the sequence" in text
        or "after analyzing the sequence" in text
    )

    return (
        has_all_choices
        and has_analysis_instruction
    )


tsqa_like_indices = [
    index
    for index, text in enumerate(sft["text"])
    if is_tsqa_text(text)
]

print("SFT rows:", len(sft))
print("TSQA-like rows:", len(tsqa_like_indices))
print(
    "TSQA-like ratio:",
    len(tsqa_like_indices) / len(sft),
)