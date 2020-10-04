# %%
#  steps for preparing data:

# Data selection
# Data processing - formatting, cleaning, sampling
# Data transformation - put cleaned data in simpler format
#%%
import pandas as pd
# %%
file_path = "iris.csv"
iris_df = pd.read_csv(file_path)
iris_df
# %%
# data seletion
# drop class
new_iris_df = iris_df.drop(['class'], axis=1)
new_iris_df
# %%
# data processing - grabbing column list to copy 
# new order into new DF
list(new_iris_df)
# %%
# data processing - setting new column order
new_iris_df = new_iris_df[['sepal_length', 'petal_length', 'sepal_width', 'petal_width']]
list(new_iris_df)
# %%
# data transformation - outputting to a simpler format
output_file_path = 'new_iris_data.csv'
new_iris_df.to_csv(output_file_path, index=False)