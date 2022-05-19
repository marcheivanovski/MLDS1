import pandas as pd
import numpy as np

#df = pd.read_csv('housing2r.csv', sep=',')
df = pd.read_csv('housing3.csv', sep=',')

for i in df.columns[:-1]:
    df[i] = (df[i] - df[i].mean()) / df[i].std()

for i in df.columns[:-1]:
    print(f"Feature has min {df[i].min():.2f} and max {df[i].max():.2f} ({i})")


class_mapping = {'C1': 0, 'C2': 1}
df['Class'] = df['Class'].map(class_mapping)
#print(np.unique(df["Class"]))

df.to_csv('housing3_standardized.csv', sep=',', index = False)