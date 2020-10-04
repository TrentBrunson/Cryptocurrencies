#%%
# Initial imports
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import hvplot.pandas
# %%
# Load data
file_path = "Resources/shopping_data_cleaned.csv"
df_shopping = pd.read_csv(file_path)
df_shopping
# %%
df_shopping.hvplot.scatter(x="AnnualIncome", y="SpendingScore")
# %%
# Function to cluster and plot dataset
# input is any dataframe and number of clusters
def test_cluster_amount(df, clusters):
   model = KMeans(n_clusters=clusters, random_state=5)   
   model
   # Fitting model
   model.fit(df)
   # Add a new class column to df_iris
   df["class"] = model.labels_
# %%
test_cluster_amount(df_shopping, 2)
df_shopping.hvplot.scatter(x="AnnualIncome", y="SpendingScore", by='class')

# %%
fig = px.scatter_3d(
	df_shopping,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()
# %%
# call function with 3 clusters
test_cluster_amount(df_shopping, 3)
df_shopping.hvplot.scatter(x="AnnualIncome", y="SpendingScore", by='class')
# %%
# call function with 4 clusters
test_cluster_amount(df_shopping, 4)
df_shopping.hvplot.scatter(x="AnnualIncome", y="SpendingScore", by='class')

# create 3-D graph
fig = px.scatter_3d(
	df_shopping,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()
# %%
# call function with 5 clusters
test_cluster_amount(df_shopping, 5)
df_shopping.hvplot.scatter(x="AnnualIncome", y="SpendingScore", by='class')

# create 3-D graph
fig = px.scatter_3d(
	df_shopping,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()
# %%
# call function with 5 clusters
test_cluster_amount(df_shopping, 5)
df_shopping.hvplot.scatter(x="AnnualIncome", y="SpendingScore", by='class')

# create 3-D graph
fig = px.scatter_3d(
	df_shopping,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()
# %%
# call function with 7 clusters
test_cluster_amount(df_shopping, 7)
df_shopping.hvplot.scatter(x="AnnualIncome", y="SpendingScore", by='class')

# create 3-D graph
fig = px.scatter_3d(
	df_shopping,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()
# %%
# Let's get ridiculous...
# call function with 100 clusters
test_cluster_amount(df_shopping, 100)
df_shopping.hvplot.scatter(x="AnnualIncome", y="SpendingScore", by='class')

# create 3-D graph
fig = px.scatter_3d(
	df_shopping,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()