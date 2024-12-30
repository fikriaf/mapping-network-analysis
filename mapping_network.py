import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import Point
import networkx as nx
import random
import sampel
import tower

# Membaca shapefile dan data lain
shapefile_path = "shape_Banten/indonesia_Province_level_1.shp"
world = gpd.read_file(shapefile_path)
banten = world[world["shape1"] == 'Banten']
banten_geom = banten.geometry.iloc[0]

provider = "Telkomsel"

# ---------------------------------------------------------------------------------------------
# Mengumpulkan Data Tower
print("[Mengumpulkan Lokasi Tower...]")
tower_positions = tower.get_tower()

print("[Mengumpulkan Data Tower]")
data_tower = tower.towers
data_dBm_tower = [int(x['max_signal_rsrp'].replace(' dBm', '')) for x in data_tower]

# Mengumpulkan Data User
print(f"[Mengumpulkan Data Pengguna Provider {provider}...]")
data_sinyal = sampel.data(tower_positions, 1000)


# ---------------------------------------------------------------------------------------------
# Buat GeoDataFrame untuk sinyal
df_sinyal = pd.DataFrame(data_sinyal)
geometry = [Point(xy) for xy in zip(df_sinyal["longitude"], df_sinyal["latitude"])]
geo_df_sinyal = gpd.GeoDataFrame(df_sinyal, geometry=geometry)


# ---------------------------------------------------------------------------------------------
# Algoritma NetworkX
G = nx.Graph()

# Menambahkan node untuk tower
for i, (lon, lat) in enumerate(tower_positions):
    G.add_node(f"Tower_{i}", pos=(lon, lat), type="tower")
    
# Menambahkan node untuk sinyal
for i, row in df_sinyal.iterrows():
    G.add_node(f"Signal_{i}", pos=(row["longitude"], row["latitude"]), 
               type="signal", signal_strength=row["signal_strength"])
    
# ---------------------------------------------------------------------------------------------
# Algoritma Nearest Neighbor
edges = []
for i, row in df_sinyal.iterrows():
    signal_point = Point(row["longitude"], row["latitude"])
    closest_tower = None
    min_distance = float("inf")
    
    for j, (lon, lat) in enumerate(tower_positions):
        tower_point = Point(lon, lat)
        distance = signal_point.distance(tower_point)
        if distance < min_distance:
            min_distance = distance
            closest_tower = f"Tower_{j}"
    if closest_tower is not None:
        edges.append((f"Signal_{i}", closest_tower, min_distance))

G.add_weighted_edges_from(edges)
# ---------------------------------------------------------------------------------------------

mst = nx.minimum_spanning_tree(G)
fig, ax = plt.subplots(figsize=(12, 10))
banten.plot(ax=ax, color="lightgrey", edgecolor="black")
ax.set_title(f"Persebaran Sinyal Telkomsel di Banten Berbasis GIS\nMenggunakan Algoritma Network Analisys dan Nearest Neighbor")

# ---------------------------------------------------------------------------------------------
# Plot tower
for lon, lat in tower_positions:
    ax.scatter(lon, lat, color="blue", s=10, label="Tower Provider Telkomsel")
# Plot sinyal
for i, row in df_sinyal.iterrows():
    lon = row["longitude"]
    lat = row["latitude"]
    distances = [Point(t).distance(Point(lon, lat)) for t in tower_positions]
    min_distance = min(distances)
    get_dBm_tower = data_dBm_tower[distances.index(min_distance)]

    color = "red" if row["signal_strength"] <= get_dBm_tower else \
            "yellow" if row["signal_strength"] <= get_dBm_tower+16 else "green"
    ax.scatter(row["longitude"], row["latitude"], color=color, s=10)

# ---------------------------------------------------------------------------------------------
# Visualisasi jaringan dengan NetworkX
print("[Menampilkan Visualisasi...]")
pos = nx.get_node_attributes(G, "pos")
nx.draw_networkx_edges(mst, pos, ax=ax, alpha=0.3, edge_color="gray")

xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Label kiri
ax.text(xlim[0] - 10, 0.5, 'Label Kiri: Persebaran Sinyal', fontsize=12, ha='center', va='center', rotation=90)

# Label kanan
ax.text(xlim[1] + 10, 0.5, 'Label Kanan: Menggunakan Algoritma MST', fontsize=12, ha='center', va='center', rotation=-90)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'Tower Provider {provider} (100 tower)',
           markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='User Sinyal Kuat (RSRP >= -80dBm)',
           markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='User Sinyal Sedang (-96dBm < RSRP < -80dBm)',
           markerfacecolor='yellow', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='User Sinyal Lemah (RSRP <= -96dBm)',
           markerfacecolor='red', markersize=10),
]
plt.legend(handles=legend_elements, loc="upper left")
plt.show()
