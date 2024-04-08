#!/usr/bin/python

######################################################
#
# Requirements
#
######################################################

import os
import sys
import getopt
import traceback
import pdb
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator
from matplotlib.path import Path
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import xarray as xr
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point


######################################################
#
# Function gen_landsea_mask
#
######################################################

def gen_landsea_mask(dataset, resolution, filename):

    ######################################################
    #
    # Define the regular grid and build land-sea mask
    #
    ######################################################

    # Extract data
    print("[gen_landsea_mask] === Reading lat and lon")
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    
    # Define the regular grid
    print("[gen_landsea_mask] === Creating the regular grid")
    lon_min, lon_max, lat_min, lat_max = np.min(lon), np.max(lon), np.min(lat), np.max(lat)
    lon_step, lat_step = resolution, resolution
    new_lons = np.arange(lon_min, lon_max, lon_step)
    new_lats = np.arange(lat_min, lat_max, lat_step)
    lon_reg, lat_reg = np.meshgrid(new_lons, new_lats)
    grid_values = np.full(lon_reg.shape, np.nan)

    # Flatten the regular grid coordinates for querying
    points_regular_flat = np.column_stack((lon_reg.ravel(), lat_reg.ravel()))

    
    ######################################################
    #
    # Fill triangles in land sea mask
    #
    ######################################################    

    # Iterate over triangles
    print("[landsea] === Iterating over nv items...")

    nv = dataset.variables['nv'][:].T - 1
    triangles = []
    for i in range(nv.shape[0]):
        
        # Get the vertices of the current triangle and convert them to shapely Polygon
        vertices = [(lon[nv[i, j]], lat[nv[i, j]]) for j in range(3)]
        triangles.append(Polygon(vertices))
        
    # Create a GeoDataFrame to store triangles
    print("[landsea] === Storing triangles in a GeoDataFrame...")    
    gdf_triangles = gpd.GeoDataFrame(geometry=triangles)
        
    # Converts the grid points of the regular grid to Shapely Point objects
    print("[landsea] === Converting grid points in Shapely Points...")
    points = [Point(p) for p in points_regular_flat]

    # Create a GeoDataFrame for the Grid Points (and init to nan)
    print("[landsea] === Storing grid points in a GeoDataFrame...")    
    gdf_points = gpd.GeoDataFrame(geometry=points)
    gdf_points['value'] = np.nan

    # Find grid points belonging to the triangles through the intersection (join) of gdf
    print("[landsea] === Filling the triangles...")    
    inner_points = gpd.sjoin(gdf_points, gdf_triangles, how="inner", predicate='within')
    gdf_points.loc[inner_points.index, 'value'] = 1
   
    
    ######################################################
    #
    # Output to NetCDF File
    #
    ######################################################

    # Prepare data to be stored on NetCDF file
    print("[landsea] === Preparing data for output to NetCDF file...")    
    data = {
        'lat': gdf_points['geometry'].y,
        'lon': gdf_points['geometry'].x,
        'value': gdf_points['value']
    }
    gdf = gpd.GeoDataFrame(data, columns=['lat', 'lon', 'value'])
    
    # Convert the pivot dataframe to a pd dataframe
    df = pd.DataFrame(gdf)

    # Create a pivot table (useful for conversion)
    pivot_table = df.pivot(index='lat', columns='lon', values='value')

    # Convert the pivot table to a numpy array
    land_sea_mask = pivot_table.to_numpy()

    # Save original coordinates
    latitudes = pivot_table.index.to_numpy()
    longitudes = pivot_table.columns.to_numpy()
    
    # Salva il Dataset come file NetCDF
    print("[landsea] === Generating NetCDF file...")    
    index = np.arange(len(gdf_points))
    ds = xr.Dataset(
        {
            'nav_lon': (('lat')),
            'nav_lat': (('lon')),
            'land_sea_mask': (('lat', 'lon'), land_sea_mask)
        },
            coords={
                'lon': ('lon', longitudes),
                'lat': ('lat', latitudes),                
            }
    )
    
    # Salva il Dataset come file NetCDF
    ds.to_netcdf(mask_filename)
    
    print("[landsea] === Land sea mask file ready!")

    

    

# ######################################################
# #
# # Preliminary data extraction
# #
# ######################################################






# # Assuming the first time step and depth layer
# # TODO:
# # make the script able to iterate over the depth layers
# myvar = dataset.variables[var_to_extract][0, 0, :]




# # initialize an array to hold the interpolated salinity data
# # New shape will be (time, depth_levels, node)
# myvar_ex = dataset.variables[var_to_extract]
# salinity_interpolated = np.zeros((myvar_ex.shape[0], desired_depth_levels, myvar_ex.shape[2]))

# # Example: Interpolating data for each node
# h = dataset.variables['h']
# original_data = dataset.variables[var_to_extract][0,:,:]
# interpolated_myvar = np.full((original_data.shape[1], original_data.shape[0]), np.nan)

# for node in range(len(h)):
    
#     # Original depth levels for this node, from 0 to h[node]
#     original_depths = np.linspace(0, h[node], original_data.shape[1])

#     # Data values for this node
#     data_values = original_data[:, node]

#     # Create interpolation function
#     interp_func = interp1d(depths_scaled[:, node], data_values, bounds_error=False, fill_value="extrapolate")
    
#     # Interpolate to new depth levels
#     interpolated_myvar[node, :] = interp_func(desired_depth_levels)




# ######################################################
# #
# # Interpolate the variable requested by the user
# #
# ######################################################

# # Debug print
# print("[landsea] --- Interpolate variable %s on the regular grid" % var_to_extract)

# # Interpolate salinity data onto the regular grid
# myvar_interpolated = griddata(points_unstructured, myvar, (lon_reg, lat_reg), method='linear')

# # Initialize a new array for the masked salinity data
# myvar_masked = np.full(myvar_interpolated.shape, np.nan)

# # Fill the masked salinity array with interpolated values where the land-sea mask is True (water)
# myvar_masked[land_sea_mask_reg] = myvar_interpolated[land_sea_mask_reg]






######################################################
#
# Main
#
######################################################

if __name__ == "__main__":

    ######################################################
    #
    # Input params management
    #
    ######################################################
    
    # read input params
    options, remainder = getopt.getopt(sys.argv[1:], 'i:o:r:v:', ['input=', 'output=', 'resolution=', 'variables='])

    # parse input params
    for opt, arg in options:
        
        if opt in ('-o', '--output'):
            output_filename = arg
            print("[main] === Output file set to: %s" % output_filename)

        elif opt in ('-i', '--input'):
            input_filename = arg
            print("[main] === Input file set to: %s" % input_filename)

        elif opt in ('-r', '--resolution'):
            resolution_meters = arg
            resolution_degrees = round(float(resolution_meters) / 111319, 6)
            print("[main] === Output resolution set to: %s m (%s deg E)" % (resolution_meters, round(resolution_degrees, 6)))

        elif opt in ('-v', '--variables'):
            variables = arg.split(":")
            print("[main] === Variables of interest are: %s" % " ".join(variables))            
            
        else:
            print("[main] === Unrecognized option %s. Will be ignored." % opt)


    ######################################################
    #
    # Input file opening
    #
    ######################################################

    # open dataset
    print("[main] === Opening dataset...")
    input_dataset = nc.Dataset(input_filename)
    
    
    ######################################################
    #
    # Generation of land sea mask
    #
    ######################################################

    # generate the land sea mask
    print("[main] === Invoking gen_landsea_mask()")
    output_path, output_basename = os.path.split(output_filename)
    mask_filename = "%s/mask_%s" % (output_path, output_basename)
    gen_landsea_mask(input_dataset, resolution_degrees, mask_filename)



    

    ######################################################
    #
    # End of business
    #
    ######################################################

    # End of business
    print("[main] === EOB. Bye...")    
