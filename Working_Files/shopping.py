#%%
import pandas as pd
# %%
file_path = 'Resources/shopping_data.csv'
shopping_df = pd.read_csv(file_path, encoding='ISO-8859-1')
shopping_df
#%%
# **************
# DATA SELECTION
# **************
# %%
# columns
list(shopping_df)
# %%
# columns
shopping_df.columns
# %%
# list  DF data types
shopping_df.dtypes
# %%
# see number of rows and columns
shopping_df.shape
# %%
# find null values
for column in shopping_df.columns:
    print(f"Column \"{column}\" has {shopping_df[column].isnull().sum()} null values.")
# %%
# drop rows will nulls
shopping_df = shopping_df.dropna()
shopping_df.shape
# %%
# check for duplicates
print(f"Duplicate entries: {shopping_df.duplicated().sum()}")
# %%
# customer ID column doesn't provide anything for the model
# it's just categorical data - removing
shopping_df.drop(columns=['CustomerID'], inplace= True)
shopping_df
# %%
# ***************
# DATA PROCESSING
# ***************
# handle nulls, convert all categories to numerical data
# scale the values, normalize
# %%
# change card member string to binary, yes = 1, all else to 0
# transform the string columnnnnn
def change_string(member):
    if member == "Yes": # running for first time, change "Yes" to 1
        return 1
    elif member == 1: # if already run, just keep same value
        return 1
    else: # anythinge else just return 0
        return 0


# update DF by calling change_string F(x)
shopping_df["Card Member"] = shopping_df["Card Member"].apply(change_string)
shopping_df
# %%
shopping_df.dtypes
# %%
# rescale the annual income data to be in line with scales of other columns
shopping_df["Annual Income"] = shopping_df["Annual Income"]/1000
shopping_df
# %%
# remove spaces
shopping_df.columns = shopping_df.columns.str.replace(" ", "")
shopping_df.columns = shopping_df.columns.str.replace('[^a-zA-Z]', '')
shopping_df
# %%
shopping_df
# %%

# %%
# ***************
# DATA TRANSFORMATION
# ***************
# %%
