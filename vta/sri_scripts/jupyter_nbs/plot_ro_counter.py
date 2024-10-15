import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.read_csv("ro_out_working_10_resnets.csv")

df1.plot(subplots=True, figsize=(100,10))

plt.show()