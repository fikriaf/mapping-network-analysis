import geopandas as gpd
from shapely.geometry import Point
import random

shapefile_path = "shape_Banten/indonesia_Province_level_1.shp"
world = gpd.read_file(shapefile_path)

# Memilih wilayah Banten
banten = world[world["shape1"] == 'Banten']
banten_geom = banten.geometry.iloc[0]
def random_tower(jumlah):
    tower_positions = []
    x = 0
    for _ in range(jumlah):
        while True:
            lat = random.uniform(banten.bounds.miny.min(), banten.bounds.maxy.max())
            lon = random.uniform(banten.bounds.minx.min(), banten.bounds.maxx.max())
            point = Point(lon, lat)
            if banten_geom.contains(point):
                x+=1
                tower_positions.append((lon, lat))
                print(f"Tower {x} : lat {lat} long {lon}")
                break
    print()
    return tower_positions