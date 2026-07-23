import pandas as pd

path = "workspace/outputs/analysis/mamba-2.8b-pre-r/1000/chattime_embedding_pca.csv"
df = pd.read_csv(path)

print("\nLargest absolute pc1")
print(df.loc[df["pc1"].abs().nlargest(20).index].to_string(index=False))

print("\nLargest absolute PC2")
print(df.loc[df["pc2"].abs().nlargest(20).index].to_string(index=False))