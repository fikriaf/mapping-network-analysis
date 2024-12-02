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

provider = "X"
lokasi_tower = [
    (106.39588367887067, -6.109830593420505),
    (106.42711714995869, -6.232778477997128),
    (106.4747400809301, -6.2574895930647365),
    (106.5099226690555,  -6.2222056665356),
    (106.62083648362552, -6.186709674342691),
    (106.7264325125655,  -6.275067589565951),
    (106.77285905125615, -6.304976580692439),
    (106.19161127463387, -6.6319367876668505),
    (106.39061922684432, -6.609428575868837),
    (106.49012320294953, -6.766964350087271)
]


# ---------------------------------------------------------------------------------------------
# Simulasi posisi tower provider X dan sampel pengguna provider X
print("[Mengumpulkan Lokasi Tower...]")
tower_positions = tower.random_tower(10)

print(f"[Mengumpulkan Data Pengguna Provider {provider}...]")
data_sinyal = sampel.data(tower_positions, 1000)


# ---------------------------------------------------------------------------------------------
# Buat GeoDataFrame untuk sinyal
df_sinyal = pd.DataFrame(data_sinyal)
geometry = [Point(xy) for xy in zip(df_sinyal["longitude"], df_sinyal["latitude"])]
geo_df_sinyal = gpd.GeoDataFrame(df_sinyal, geometry=geometry)


# ---------------------------------------------------------------------------------------------
# Membuat graf jaringan dengan NetworkX
G = nx.Graph()

# Menambahkan node untuk tower
for i, (lon, lat) in enumerate(tower_positions):
    G.add_node(f"Tower_{i}", pos=(lon, lat), type="tower")
    
# Menambahkan node untuk sinyal
for i, row in df_sinyal.iterrows():
    G.add_node(f"Signal_{i}", pos=(row["longitude"], row["latitude"]), 
               type="signal", signal_strength=row["signal_strength"])

# Menambahkan edge berdasarkan jarak, hanya menghubungkan ke tower terdekat
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

# Menambahkan edges ke graf
G.add_weighted_edges_from(edges)
# Membuat Minimum Spanning Tree (MST) dari graf
mst = nx.minimum_spanning_tree(G)
# Visualisasi peta
fig, ax = plt.subplots(figsize=(12, 10))
banten.plot(ax=ax, color="lightgrey", edgecolor="black")
ax.set_title(f"Persebaran Sinyal Pengguna {provider} di Banten Berbasis Algoritma Network Analisys")
# Plot tower
for lon, lat in tower_positions:
    ax.scatter(lon, lat, color="blue", s=100, label="Tower Provider X")
# Plot sinyal
for i, row in df_sinyal.iterrows():
    color = "green" if row["signal_strength"] >= 80 else \
            "yellow" if row["signal_strength"] >= 50 else "red"
    ax.scatter(row["longitude"], row["latitude"], color=color, s=50)

# ---------------------------------------------------------------------------------------------
# Visualisasi jaringan dengan NetworkX
print("[Menampilkan Visualisasi...]")
pos = nx.get_node_attributes(G, "pos")
nx.draw_networkx_edges(mst, pos, ax=ax, alpha=0.3, edge_color="gray")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'Tower Provider {provider}',
           markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Sinyal Kuat',
           markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Sinyal Sedang',
           markerfacecolor='yellow', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Sinyal Lemah',
           markerfacecolor='red', markersize=10),
]
plt.legend(handles=legend_elements, loc="upper left")
plt.show()
