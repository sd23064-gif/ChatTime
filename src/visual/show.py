import pandas as pd
import numpy as np

df = pd.read_csv("./outputs/chattime_embedding_vis/chattime_embedding_pca_coordinates.csv")

df["sign_value"] = np.sign(df["value"])
df["abs_value"] = df["value"].abs()

cols = ["pc1", "pc2", "value", "abs_value", "sign_value"]

corr = df[cols].corr()

print(corr)