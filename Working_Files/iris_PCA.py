#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hvplot.pandas
# %%
file_path = "Resources/new_iris_data.csv"
df_iris = pd.read_csv(file_path)
df_iris
# %%
# PCA needs scaled data
# standardize data with StandardScaler
iris_scaled = StandardScaler().fit_transform(df_iris)
print(iris_scaled[0:12])
# %%
iris_scaled.shape
# %%
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
# use elbow curve with generate principle components
# Find the best value for K
inertia = []
k = list(range(1, 11))

# Calculate the inertia for the range of K values
for i in k:
	km = KMeans(n_clusters=i, random_state=0)
	km.fit(df_iris_pca)
	inertia.append(km.inertia_)

# Create the elbow curve
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", xticks=k, title="Elbow Curve")
# %%
# use principal components data with K-means algorithm 
# and with K-value = 3
# Initialize the K-means model
model = KMeans(n_clusters=3, random_state=0)

# Fit the model
model.fit(df_iris_pca)

# Predict clusters
predictions = model.predict(df_iris_pca)

# Add the predicted class columns
df_iris_pca["class"] = model.labels_
df_iris_pca.head()
# %%
# plot clusters
df_iris_pca.hvplot.scatter(
	x="principal component 1",
	y="principal component 2",
	hover_cols=["class"],
	by="class",
)
# %%
