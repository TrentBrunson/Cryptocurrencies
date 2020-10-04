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
df_shopping.head(10)
# %%
# create empty list to hold inertia values
inertia = []
k = list(range(1, 11))

# Looking for the best K
# Calculate the inertia for the range of K values
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(df_shopping)
    inertia.append(km.inertia_)
inertia
# %%
# Create elbow curve using HVPlot
# create DF to store K-values with respective inertia values
# Define a DataFrame to plot the Elbow Curve using hvPlot
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)
# %%
# create K-means functions to reuse K-means cluster
def get_clusters(k, data):   
    # Create a copy of the DataFrame   
    data = data.copy()       
    # Initialize the K-Means model   
    model = KMeans(n_clusters=k, random_state=0)   
    # Fit the model   
    model.fit(data)   
    # Predict clusters   
    predictions = model.predict(data)   
    # Create return DataFrame with predicted clusters   
    data["class"] = model.labels_   
    return data
# %%
five_clusters = get_clusters(5, df_shopping)
five_clusters.head()
#%%
six_clusters = get_clusters(6, df_shopping)
six_clusters.head()
# %%
# plotting 2D scatter plot for Annual Income and Spending Score
five_clusters.hvplot.scatter(x="AnnualIncome", y="SpendingScore", by='class')
# %%
# visualize with 5 clusters
# Plot the 3D-scatter for Annual Income Spending Score and Age
fig = px.scatter_3d(
	five_clusters,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()
#%%
# plotting 2D scatter plot for Annual Income and Spending Score
six_clusters.hvplot.scatter(x="AnnualIncome", y="SpendingScore", by='class')
# %%
# visualize with 6 clusters
# Plot the 3D-scatter for Annual Income Spending Score and Age
fig = px.scatter_3d(
	six_clusters,
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
