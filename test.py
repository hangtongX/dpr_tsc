import torch
import pandas as pd

data = pd.DataFrame([[1,2,3],[2,1,3],[3,1,2]],columns=['a','b','c'])
print(data.b.argsort())
print(data)