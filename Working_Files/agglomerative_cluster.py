#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import hvplot.pandas
# %%
file_path = "Resources/new_iris_data.csv"
df_iris = pd.read_csv(file_path)
df_iris
# %%
# apply PCA to reduce feature set from 4 to 2
# PCA needs scaled data
# standardize data with StandardScaler
iris_scaled = StandardScaler().fit_transform(df_iris)
print(iris_scaled[0:12])

# after standardizing, use PCA to reduce features
# Initialize PCA model
pca = PCA(n_components=2)

# Get two principal components for the iris data.
iris_pca = pca.fit_transform(iris_scaled)

# transform PCA data into DF
df_iris_pca = pd.DataFrame(
    data=iris_pca, columns=['principal component 1', 'principal component 2']
)
df_iris_pca.head(11)
# %%
# fetch explained variance
pca.explained_variance_ratio_
# %%
# start hierachal clustering; start with dendrogram
import plotly.figure_factory as ff

# pass a color_threshold of 0 to make all the branches the same color
# Create the dendrogram
fig = ff.create_dendrogram(df_iris_pca, color_threshold=0)
fig.update_layout(width=800, height=500)
fig.show()

# looking at the dendrogram, the iris dataset contains three clusters
# The cutoff will be set at five to obtain three clusters
# %%
# set up agg model; use n_clusters set to 3 as determined
# in previous step
agg = AgglomerativeClustering(n_clusters=3)
model = agg.fit(df_iris_pca)

# add new class column to iris DF
df_iris_pca['class'] = model.labels_
df_iris_pca
# %%
# plot results of hierarchal clustering algorithm
df_iris_pca.hvplot.scatter(
    x="principal component 1",
	y="principal component 2",
	hover_cols=["class"],
	by="class"
)
# %%
