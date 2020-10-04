#%%
import pandas as pd
import plotly.express as px
import hvplot.pandas
from sklearn.cluster import KMeans
# %%
file_path = 'Resources/new_iris_data.csv'
df_iris = pd.read_csv(file_path)
df_iris
# %%
# initialize K starting centroids
# initializing model with K=3 since already know there are three classes of iris plants
model = KMeans(n_clusters=3, random_state=5) # random state set for reproducible purposes
model
# %%
# fitting model
model.fit(df_iris)
# %%
# get predictions
predictions = model.predict(df_iris)
print(predictions)
# %%
# add new column to DF with predicted classes
df_iris['class'] = model.labels_
df_iris
# %%
# visualize results
# incorporate DF's new class column into graph
df_iris.hvplot.scatter(x='sepal_length', y='sepal_width', by='class')
# %%
# creating 3D plot to see if visualization is different
fig = px.scatter_3d(
    df_iris,
    x='petal_width',
    y='sepal_length',
    z='petal_length',
    color='class',
    symbol='class',
    size='sepal_width',
    width=800
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()
# %%
