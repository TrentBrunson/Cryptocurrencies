#%%
# Initial imports
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import hvplot.pandas
# %%
# Loading data
file_path = "Resources/new_iris_data.csv"
df_iris = pd.read_csv(file_path)
df_iris
# %%
# create empty list to hold inertia values
inertia = []
k = list(range(1, 11))
# %%
# Looking for the best K
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(df_iris)
    inertia.append(km.inertia_)
inertia
# %%
# create DF to store K-values with respective inertia values
# Define a DataFrame to plot the Elbow Curve using hvPlot
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)
# %%
