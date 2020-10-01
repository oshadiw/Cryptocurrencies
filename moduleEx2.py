#%%
# Initial imports
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import hvplot.pandas

# %%
# Load data
file_path = "/Users/oshadi/Desktop/Analysis Projects/Cryptocurrencies/module examples/shopping_data_cleaned.csv"
df_shopping = pd.read_csv(file_path)
df_shopping.head(10)

# %%
df_shopping.hvplot.scatter(x="Annual Income", y="Spending Score (1-100)")

# %%
# Function to cluster and plot dataset
def test_cluster_amount(df, clusters):
    model = KMeans(n_clusters=clusters, random_state=5)   
    model
    model.fit(df)
    df["class"] = model.labels_

# %%
test_cluster_amount(df_shopping, 2)
df_shopping.hvplot.scatter(x="Annual Income", y="Spending Score (1-100)", by="class")

# %%
fig = px.scatter_3d(
	df_shopping,
x="Annual Income",
	y="Spending Score (1-100)",
	z="Age",
color="class",
	symbol="class",
	width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()

# %%
file_path = "/Users/oshadi/Desktop/Analysis Projects/Cryptocurrencies/module examples/new_iris_data.csv"
df_iris = pd.read_csv(file_path)
df_iris.head(10)

# %%
inertia = []
k = list(range(1, 11))

# %%
# Looking for the best K
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(df_iris)
    inertia.append(km.inertia_)

# %%
# Define a DataFrame to plot the Elbow Curve using hvPlot
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)


# %%
inertia = []
k = list(range(1, 11))
# Calculate the inertia for the range of K values
for i in k:
   km = KMeans(n_clusters=i, random_state=0)
   km.fit(df_shopping)
   inertia.append(km.inertia_)

# %%
file_path = "/Users/oshadi/Desktop/Analysis Projects/Cryptocurrencies/module examples/new_iris_data.csv"
df_iris = pd.read_csv(file_path)
df_iris.head(10)

# %%
inertia = []
k = list(range(1, 11))
# Calculate the inertia for the range of K values
for i in k:
   km = KMeans(n_clusters=i, random_state=0)
   km.fit(df_shopping)
   inertia.append(km.inertia_)

# %%
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)

# %%
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

# %%
six_clusters = get_clusters(6, df_shopping)
six_clusters.head()

# %%
# Plot the 3D-scatter with x="Annual Income", y="Spending Score (1-100)" and z="Age"
fig = px.scatter_3d(
    five_clusters,
    x="Age",
    y="Spending Score (1-100)",
    z="Annual Income",
    color="class",
    symbol="class",
    width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()

# %%
# Plotting the 3D-Scatter with x="Annual Income", y="Spending Score (1-100)" and z="Age"
fig = px.scatter_3d(
    six_clusters,
    x="Age",
    y="Spending Score (1-100)",
    z="Annual Income",
    color="class",
    symbol="class",
    width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()

# %%
