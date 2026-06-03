from huggingface_hub import list_repo_files

files = list_repo_files(
    repo_id="ChengsenWang/ChatTime-1-Pretrain-1M",
    repo_type="dataset"
)

print(files[:100])