import geopandas as gpd
import matplotlib.pyplot as plt

# Membaca shapefile dan data lain
shapefile_path = "shape_Banten/indonesia_Province_level_1.shp"
world = gpd.read_file(shapefile_path)

# Filter untuk provinsi Banten
banten = world[world["shape1"] == 'Banten']

# Visualisasi peta
fig, ax = plt.subplots(figsize=(12, 10))
banten.plot(ax=ax, color="lightgrey", edgecolor="black")
ax.set_title("Map of Banten Province")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
