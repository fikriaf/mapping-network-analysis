# Signal Distribution Analysis for Provider in Banten

![Signal Distribution Analysis for Provider in Banten](result/result.png)

This script is used to analyze and visualize the signal distribution of users from a provider in the Banten region, Indonesia. The process includes spatial data processing, tower position simulation, and network analysis using the Minimum Spanning Tree (MST) algorithm.

## Key Features
- **Spatial Data Processing**
- **Tower Position Simulation**
- **User Data Simulation**
- **Network Analysis**
- **Visualization**

## Requirements
Ensure the following libraries are installed:
- `geopandas`
- `pandas`
- `matplotlib`
- `networkx`
- `shapely`

To install all these libraries, use:

```bash
pip install geopandas pandas matplotlib networkx shapely
```
## Project Structure
shape_Banten/indonesia_Province_level_1.shp: Shapefile for the Indonesian region.
tower.py: Module for generating tower locations.
sampel.py: Module for simulating user and signal data.

## How to Use
Ensure all related files (shapefile, tower module, and sampel module) are in the same directory as the main script.

Run the script with Python:

```bash
python mapping_network.py
```

The signal distribution and network map will be displayed after the simulation completes.

## Visualization Results
Signal Distribution Map: Shows tower and user locations based on signal strength:
- Green: Strong signal.
- Yellow: Medium signal.
- Red: Weak signal.
- Network Graph: Displays the connection between towers and users using the MST algorithm.
