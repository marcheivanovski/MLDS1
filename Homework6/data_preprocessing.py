import pandas as pd
import numpy as np

#df = pd.read_csv('housing2r.csv', sep=',')
df = pd.read_csv('train.csv', sep=',')
df1 = pd.read_csv('test.csv', sep=',')

for i in df.columns[1:-1]:
    #print(i, df[i].mean(), df[i].std())
    df[i] = (df[i] - df[i].mean()) / df[i].std()
    df1[i] = (df1[i] - df[i].mean()) / df[i].std()


for i in df.columns[1:-1]:
    print(f"Feature has min {df[i].min():.2f} and max {df[i].max():.2f} ({i})")


#class_mapping = {'C1': 0, 'C2': 1}
#df['Class'] = df['Class'].map(class_mapping)
#print(np.unique(df["Class"]))

class_mapping = {'Class_1': 0,'Class_2': 1,'Class_3': 2,'Class_4': 3,'Class_5': 4,'Class_6': 5,'Class_7': 6,
 'Class_8': 7,'Class_9': 8}

df['target'] = df['target'].map(class_mapping)

del df['id']
del df1['id']

df.to_csv('train_standardized.csv', sep=',', index = False)
df1.to_csv('test_standardized.csv', sep=',', index = False)