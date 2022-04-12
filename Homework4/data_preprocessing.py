import pandas as pd

df = pd.read_csv('housing2r.csv', sep=',')

for i in df.columns[:-1]:
    df[i] = (df[i] - df[i].mean()) / df[i].std()

for i in df.columns[:-1]:
    print(f"Feature has min {df[i].min():.2f} and max {df[i].max():.2f} ({i})")

df.to_csv('housing2r_standardized.csv', sep=',', index = False)