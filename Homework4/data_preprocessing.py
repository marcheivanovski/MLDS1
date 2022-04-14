import pandas as pd

df = pd.read_csv('housing2r.csv', sep=',')

for i in df.columns[:-1]:
    df[i] = (df[i] - df[i].mean()) / df[i].std()

for i in df.columns[:-1]:
    print(f"Feature has min {df[i].min():.2f} and max {df[i].max():.2f} ({i})")

df.to_csv('housing2r_standardized.csv', sep=',', index = False)


'''
def load_housing():
    
    df = pd.read_csv('/kaggle/input/mlds1-hw4-housing/housing2r_standardized.csv', sep=',')
    data = df.to_numpy()
    X, y = data[:,:5], data[:,5]
    return X[:160,:], y[:160], X[160:,:], y[160:]


file = os.path.join(os.sep, "C:" + os.sep, "Users", "marko" + os.sep,
    "OneDrive" + os.sep,  "Desktop" + os.sep,  "MLDS1" + os.sep,  
    "Homework4" + os.sep, 'housing2r_standardized.csv')
df = pd.read_csv(file, sep=',')
'''