city_coordinates = [
    ("Aberdeen", 46.9754, -123.8157),
    ("Alderwood Manor", 47.8223, -122.2701),
    ("Anacortes", 48.5126, -122.6127),
    ("Arlington", 48.1989, -122.1251),
    ("Artondale", 47.3018, -122.6193),
    ("Auburn", 47.3073, -122.2285),
    ("Battle Ground", 45.7804, -122.5337),
    ("Bellevue", 47.6104, -122.2007),
    ("Bellingham", 48.7491, -122.4787),
    ("Bonney Lake", 47.1773, -122.1868),
    ("Bothell", 47.7623, -122.2054),
    ("Bremerton", 47.5673, -122.6326),
    ("Burien", 47.4704, -122.3468),
    ("Camano", 48.1726, -122.5116),
    ("Camas, Clark County", 45.5871, -122.3995),
    ("Centralia", 46.7162, -122.9543),
    ("Cheney", 47.4871, -117.5758),
    ("Cottage Lake", 47.7743, -122.0846),
    ("Covington", 47.3654, -122.1015),
    ("Des Moines", 47.4018, -122.3243),
    ("Dishman", 47.6568, -117.2902),
    ("East Renton Highlands", 47.4815, -122.1001),
    ("East Wenatchee Bench", 47.4232, -120.3103),
    ("Edgewood", 47.2507, -122.2923),
    ("Edmonds", 47.8107, -122.3774),
    ("Elk Plain", 47.0468, -122.3743),
    ("Ellensburg", 46.9965, -120.5478),
    ("Enumclaw", 47.2043, -121.9904),
    ("Everett", 47.9785, -122.2021),
    ("Federal Way", 47.3223, -122.3126),
    ("Ferndale", 48.8468, -122.5915),
    ("Five Corners, Clark County", 45.6868, -122.5737),
    ("Graham", 47.0523, -122.2943),
    ("Hoquiam", 46.9804, -123.8893),
    ("Issaquah", 47.5301, -122.0326),
    ("Kelso", 46.1468, -122.9087),
    ("Kenmore", 47.7573, -122.2443),
    ("Kennewick", 46.2112, -119.1372),
    ("Kent", 47.3809, -122.2348),
    ("Kingsgate", 47.7323, -122.1793),
    ("Kirkland", 47.6815, -122.2087),
    ("Lacey", 47.0343, -122.8232),
    ("Lake Forest Park", 47.7563, -122.2804),
    ("Lakewood, Pierce County", 47.1718, -122.5185),
    ("Lea Hill", 47.3343, -122.1804),
    ("Longview, Cowlitz County", 46.1382, -122.9382),
    ("Lynden", 48.9468, -122.4523),
    ("Lynnwood", 47.8279, -122.3052),
    ("Maple Valley", 47.3668, -122.0443),
    ("Martha Lake", 47.8507, -122.2393),
    ("Marysville", 48.0518, -122.1771),
    ("Mercer Island", 47.5707, -122.2221),
    ("Mill Creek", 47.8607, -122.2043),
    ("Monroe", 47.8554, -121.9704),
    ("Moses Lake", 47.1301, -119.2781),
    ("Mount Vernon", 48.4212, -122.3348),
    ("Mountlake Terrace", 47.7887, -122.3087),
    ("Mukilteo", 47.9443, -122.3043),
    ("North Creek", 47.8218, -122.1768),
    ("North Marysville", 48.0718, -122.1771),
    ("Oak Harbor", 48.2933, -122.6432),
    ("Olympia", 47.0379, -122.9007),
    ("Opportunity", 47.6568, -117.2393),
    ("Orchards", 45.6907, -122.5373),
    ("Parkland", 47.1418, -122.4343),
    ("Pasco", 46.2396, -119.1006),
    ("Port Angeles", 48.1181, -123.4307),
    ("Prairie Ridge", 47.1443, -122.1687),
    ("Pullman", 46.7313, -117.1796),
    ("Puyallup", 47.1854, -122.2923),
    ("Redmond", 47.6731, -122.1215),
    ("Renton", 47.4829, -122.2171),
    ("Richland", 46.2857, -119.2845),
    ("Salmon Creek", 45.7107, -122.6487),
    ("Sammamish", 47.6163, -122.0353),
    ("SeaTac", 47.4436, -122.3015),
    ("Seattle", 47.6062, -122.3321),
    ("Sedro-Woolley", 48.5033, -122.2368),
    ("Shoreline", 47.7557, -122.3415),
    ("Silverdale", 47.6443, -122.6943),
    ("South Hill", 47.1418, -122.2923),
    ("Spanaway", 47.1033, -122.4343),
    ("Spokane", 47.6588, -117.4260),
    ("Sunnyside, Yakima County", 46.3243, -120.0081),
    ("Tacoma", 47.2529, -122.4443),
    ("Toppenish", 46.3771, -120.3081),
    ("Tukwila", 47.4739, -122.2600),
    ("Tumwater", 47.0073, -122.9093),
    ("University Place", 47.2357, -122.5501),
    ("Vancouver", 45.6387, -122.6615),
    ("Vashon", 47.4473, -122.4593),
    ("Veradale", 47.6438, -117.1991),
    ("Walla Walla", 46.0646, -118.3430),
    ("Waller", 47.2083, -122.3532),
    ("Wenatchee", 47.4235, -120.3103),
    ("West Lake Stevens", 48.0157, -122.0657),
    ("West Valley", 46.5957, -120.5737),
    ("White Center", 47.5093, -122.3476),
    ("Woodinville", 47.7543, -122.1635),
    ("Yakima", 46.6021, -120.5059),
]

coordinates = [(x[0].replace(" ", "%20"), x[1], x[2]) for x in city_coordinates]


# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import griddata


# lat_range = [46.5, 49]
# lon_range = [-123, -121.5]

# cities = list(filter(lambda x:(lat_range[0] <= x[1] <= lat_range[1] and lon_range[0] <= x[2] <= lon_range[1]), cities))
# latitudes = [city[1] for city in cities]
# longitudes = [city[2] for city in cities]
# city_names = [city[0] for city in cities]
# values = [np.random.random_integers(60, 100) for _ in cities]

# # Define grid
# grid_latitude, grid_longitude = np.mgrid[min(latitudes):max(latitudes):100j, min(longitudes):max(longitudes):100j]

# # Interpolate values on the grid
# grid_values = griddata((latitudes, longitudes), values, (grid_latitude, grid_longitude), method='linear')

# # Plotting the heatmap
# plt.figure(figsize=(10, 8))
# plt.imshow(grid_values.T, extent=(min(longitudes), max(longitudes), min(latitudes), max(latitudes)), origin='lower', cmap='viridis')
# plt.colorbar(label='Values')
# plt.scatter(longitudes, latitudes, c=values, edgecolors='w', linewidths=1)
# plt.title('Heatmap with Linear Interpolation')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter

# Example list of coordinates and values
# coordinates = [
#     (46.9754, -123.8157, 10),
#     (47.8223, -122.2701, 20),
#     (48.5126, -122.6127, 30),
#     (48.1989, -122.1251, 40),
#     (47.3018, -122.6193, 50),
#     (47.3073, -122.2285, 60),
#     (45.7804, -122.5337, 70),
#     (47.6104, -122.2007, 80),
#     (48.7491, -122.4787, 90),
#     (47.1773, -122.1868, 100),
# ]

# Extracting latitude, longitude and values
# latitudes = np.array([coord[0] for coord in coordinates])
# longitudes = np.array([coord[1] for coord in coordinates])
# values = [np.random.randint(60, 100) for _ in coordinates]

# coordinates = list(zip(longitudes, latitudes))

# # Define grid
# grid_x, grid_y = np.mgrid[min(latitudes):max(latitudes):100j, min(longitudes):max(longitudes):100j]

# # Interpolate data onto grid
# grid_z = griddata(coordinates, values, (grid_x, grid_y), method='cubic')

# # Find local maxima
# neighborhood_size = 3
# local_max = maximum_filter(grid_z, size=neighborhood_size) == grid_z

# # Extract peak coordinates and values
# peak_coords = np.argwhere(local_max)
# peak_values = grid_z[local_max]

# # Convert grid indices to original coordinates
# peak_longitudes = grid_x[peak_coords[:, 0], peak_coords[:, 1]]
# peak_latitudes = grid_y[peak_coords[:, 0], peak_coords[:, 1]]

# peaks = list(zip(peak_longitudes, peak_latitudes, peak_values))
# print("Peaks:", peaks)
