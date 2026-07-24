from safetensors import safe_open

adapter_path = (
    "outputs/logs/mamba-2.8b-pretrain-r/checkpoint-8000/"
    "adapter_model.safetensors"
)

dtype_counts = {}

with safe_open(adapter_path, framework="pt", device="cpu") as file:
    for key in file.keys():
        tensor = file.get_tensor(key)
        dtype_name = str(tensor.dtype)
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

        print(
            key,
            tuple(tensor.shape),
            tensor.dtype,
        )

print("\nDtype counts:")
print(dtype_counts)