import geopandas as gpd
from shapely.geometry import Point
import random

shapefile_path = "shape_Banten/indonesia_Province_level_1.shp"
world = gpd.read_file(shapefile_path)

banten = world[world["shape1"] == 'Banten']
banten_geom = banten.geometry.iloc[0]

def data(tower_positions, jumlah_sampel):
    data_sinyal = {
        "latitude": [],
        "longitude": [],
        "signal_strength": []
    }
    for _ in range(jumlah_sampel):
        while True:
            lat = random.uniform(banten.bounds.miny.min(), banten.bounds.maxy.max())
            lon = random.uniform(banten.bounds.minx.min(), banten.bounds.maxx.max())
            point = Point(lon, lat)
            if banten_geom.contains(point):
                distances = [Point(t).distance(Point(lon, lat)) for t in tower_positions]
                min_distance = min(distances)
                if min_distance < 0.08:
                    signal_strength = 100
                elif min_distance < 0.3:
                    signal_strength = 70
                else:
                    signal_strength = 40
                
                data_sinyal["latitude"].append(lat)
                data_sinyal["longitude"].append(lon)
                data_sinyal["signal_strength"].append(signal_strength)
                print(f"[{_}] Latitude: {lat:.6f} Longitude: {lon:.6f} Signal Strength: {signal_strength}", end="\r")
                break    
    print(f"{jumlah_sampel} Pengguna Selesai Dibuat.                                                    ")
    return data_sinyal