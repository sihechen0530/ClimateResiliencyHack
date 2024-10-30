import os
import pandas as pd
from prophet import Prophet
from coordinates import coordinates
from multiprocessing import Pool
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import preprocessing

weather_data_directory = "data/weather"
park_coords_file = "data/Seattle_Parks_And_Recreation_Park_Addresses_20241030.csv"
interest_columns = [
    "Minimum Temperature",
    "Mean Temperature",
    "Maximum Temperature",
    "Total Precipitation",
]


def predict(file_path, future_dates):
    df = pd.read_csv(file_path)

    # Convert the 'date' column to datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Train and forecast each relevant column with Prophet
    location_predictions = {}
    for column in interest_columns:  # Skip 'date'
        # Prepare the data in Prophet's required format
        prophet_df = df[["date", column]].rename(columns={"date": "ds", column: "y"})

        # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(prophet_df)

        # Use only the specified date range in future dates
        future = future_dates.copy()

        # Predict future values for the specified date range
        forecast = model.predict(future)

        # Store the forecasted values
        location_predictions[column] = forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ]

    return location_predictions


def find_closest_distances(df1, df2):
    """
    Calculate the closest distance from each location in df1 to any location in df2.

    Parameters:
    - df1 (pd.DataFrame): DataFrame with columns ['latitude', 'longitude'] for locations.
    - df2 (pd.DataFrame): DataFrame with columns ['latitude', 'longitude'] for spots.

    Returns:
    - distances (pd.Series): Series of minimum distances for each location in df1.
    """
    # Ensure the necessary columns are present
    required_columns = ["latitude", "longitude"]
    for df, name in zip([df1, df2], ["df1", "df2"]):
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"{name} must contain columns: {required_columns}")

    # Calculate the minimum distance from each point in df1 to any point in df2
    distances = []
    for _, row1 in df1.iterrows():
        loc1 = (row1["latitude"], row1["longitude"])
        min_distance = min(
            geodesic(loc1, (row2["latitude"], row2["longitude"])).meters
            for _, row2 in df2.iterrows()
        )
        distances.append(min_distance)
    print(distances)
    # Return distances as a Series to append to df1 if needed
    return distances


def plot_scatter(df, x_name, y_name, name):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_name], df[y_name], color="blue", marker="o")

    # Annotate each point with its name
    for i in range(len(df)):
        plt.annotate(
            df[name][i],
            (df[x_name][i], df[y_name][i]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )

    # Set labels and title
    plt.title("Locations on a 2D Plane")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid()

    # Show the plot
    plt.show()


def plot_heatmap(df, x_name, y_name, name, value_name):
    latitudes = df[y_name]
    longitudes = df[x_name]
    values = df[value_name]
    # Plot the heatmap
    grid_latitude, grid_longitude = np.mgrid[
        min(latitudes) : max(latitudes) : 100j, min(longitudes) : max(longitudes) : 100j
    ]
    grid_values = griddata(
        (latitudes, longitudes),
        values,
        (grid_latitude, grid_longitude),
        method="linear",
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(
        grid_values.T,
        extent=(min(longitudes), max(longitudes), min(latitudes), max(latitudes)),
        origin="lower",
        cmap="viridis",
    )
    for index, row1 in df.iterrows():
        plt.annotate(
            index,
            (row1[x_name], row1[y_name]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )
    plt.colorbar(label="Values")
    plt.title("Heatmap with Linear Interpolation")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()


# 1. get predictions of weather in future dates

target_years = [2025, 2026, 2027]
future_dates = pd.concat(
    [
        pd.DataFrame({"ds": pd.date_range(f"{year}-06-01", f"{year}-09-30")})
        for year in target_years
    ]
)


# Initialize an empty dictionary to store dataframes
weather_predictions = {}

async_res = {}
# Loop through all files in the directory
with Pool() as pool:
    for filename in os.listdir(weather_data_directory):
        if not filename.endswith(".csv"):
            continue
        # Extract location name and load the CSV into a DataFrame
        location_name = filename[:-4]
        file_path = os.path.join(weather_data_directory, filename)
        async_res[location_name] = pool.apply_async(
            func=predict,
            args=(
                file_path,
                future_dates,
            ),
        )
    pool.close()
    pool.join()

for k, v in async_res.items():
    try:
        weather_predictions[k] = v.get()
    except Exception as e:
        print("Exception", e)


# 2. get distance to closest by park (or other cooling facilities)

coords_df = pd.DataFrame(coordinates, columns=["location", "latitude", "longitude"])
coords_df.set_index("location", inplace=True)
coords_df = coords_df.sort_index()
print(coords_df)

park_coordinate_columns = ["LocID", "X Coord", "Y Coord"]
park_df = pd.read_csv(park_coords_file)[park_coordinate_columns]
park_df["latitude"] = park_df["Y Coord"]
park_df["longitude"] = park_df["X Coord"]
plot_scatter(park_df, "longitude", "latitude", "LocID")


column_name = "closest_distance"
# Calculate closest distances
coords_df[column_name] = find_closest_distances(coords_df, park_df)
min_value = coords_df[column_name].min()
max_value = coords_df[column_name].max()
coords_df[f"normalized_{column_name}"] = (coords_df[column_name] - min_value) / (
    max_value - min_value
)
print(coords_df)

# 3. calculate index as 1/(normalized_mean_temperature + normalized_distance)
min_temperature = 30
max_temperature = 90

max_temperature_normalized = []
for location in weather_predictions:
    max_temperature_mean = weather_predictions[location][interest_columns[2]][
        "yhat"
    ].mean()
    mean_normalized = (max_temperature_mean - min_temperature) / (
        max_temperature - min_temperature
    )
    max_temperature_normalized.append([location, mean_normalized])

max_temperature_normalized_df = pd.DataFrame(
    max_temperature_normalized, columns=["location", "temperature"]
)
max_temperature_normalized_df.set_index("location", inplace=True)
max_temperature_normalized_df = max_temperature_normalized_df.sort_index()
merged_df = coords_df.join(max_temperature_normalized_df)
merged_df["heat_index"] = (
    merged_df["temperature"] + merged_df["normalized_closest_distance"]
)
plot_heatmap(merged_df, "longitude", "latitude", "location", "heat_index")

# 4. get topk and run k means
topK = 50
latitudes = np.array(merged_df["latitude"])
longitudes = np.array(merged_df["longitude"])
values = np.array(merged_df["heat_index"])

# Define grid
grid_latitude, grid_longitude = np.mgrid[
    min(latitudes) : max(latitudes) : 100j, min(longitudes) : max(longitudes) : 100j
]

# Interpolate data onto grid
grid_values = griddata(
    (latitudes, longitudes), values, (grid_latitude, grid_longitude), method="linear"
)

plt.figure(figsize=(10, 8))
plt.imshow(
    grid_values.T,
    extent=(min(longitudes), max(longitudes), min(latitudes), max(latitudes)),
    origin="lower",
    cmap="viridis",
)
plt.colorbar(label="Values")
x = list(reversed(sorted(zip(longitudes, latitudes, values), key=lambda x: x[2])))
x = x[:topK]
plt.scatter(
    [xx[0] for xx in x],
    [xx[1] for xx in x],
    c=[xx[2] for xx in x],
    edgecolors="w",
    linewidths=1,
)
plt.title("Heatmap with Linear Interpolation")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

# 5. run k means
kMeans = 4
coordinates = [(xx[0], xx[1]) for xx in x]
X = np.array(coordinates)
kmeans = KMeans(n_clusters=kMeans, random_state=0).fit(X)
y_kmeans = kmeans.predict(X)

centers = kmeans.cluster_centers_

# Step 5: Visualize the results
plt.figure(figsize=(10, 8))
# plt.imshow(
#     grid_values.T,
#     extent=(min(longitudes), max(longitudes), min(latitudes), max(latitudes)),
#     origin="lower",
#     cmap="viridis",
# )
# plt.colorbar(label="Values")
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', marker='o')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title("K-means Clustering")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend()
plt.grid()
plt.show()

