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
# Function merge_files
#
######################################################

def merge_files(merge_list, filename):

    datasets = []
    for m in merge_list:
        print("[merge_files] === Opening %s" % m)
        datasets.append(xr.open_dataset(m))

    # merge the datasets
    print("[merge_files] === Merging...")
    ds_merged = xr.merge(datasets, compat='override')
    
    # Salva il dataset unito in un nuovo file NetCDF
    print("[merge_files] === Saving to %s" % filename)    
    ds_merged.to_netcdf(filename)

        
    
######################################################
#
# Function gen_bathymetry
#
######################################################

def gen_bathymetry(dataset, resolution, filename, mask):

    # extract lat and lon data
    print("[gen_bathymetry] === Reading lat and lon")
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    
    # define the regular grid
    print("[gen_bathymetry] === Creating the regular grid")
    lon_min, lon_max, lat_min, lat_max = np.min(lon), np.max(lon), np.min(lat), np.max(lat)
    lon_step, lat_step = resolution, resolution
    new_lons = np.arange(lon_min, lon_max, lon_step)
    new_lats = np.arange(lat_min, lat_max, lat_step)
    lon_reg, lat_reg = np.meshgrid(new_lons, new_lats)
    grid_values = np.full(lon_reg.shape, np.nan)

    # map input data on the grid
    points = np.column_stack((lon.flatten(), lat.flatten()))
    values = dataset.variables['h']

    # interpolate
    grid_values = griddata(points, values, (lon_reg, lat_reg), method='linear') 
        
    # mask
    bool_mask = np.isnan(mask)
    grid_values[bool_mask] = np.nan
    
    # generate output file
    print("[gen_bathymetry] === Generating NetCDF file...")    
    ds = xr.Dataset(
        {
            'bathymetry': (('lat', 'lon'), grid_values)
        },
            coords={
                'lon': ('lon', new_lons),
                'lat': ('lat', new_lats),                
            }
    )
    ds.to_netcdf(filename)    
    print("[gen_bathymetry] === Bathymetry file ready!")

    
######################################################
#
# Function gen_4dvar
#
######################################################

def gen_4dvar(dataset, resolution, filename, varname, mask):

    # Extract data
    print("[gen_4dvar] === Reading lat and lon")
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    
    # Define the regular grid
    print("[gen_4dvar] === Creating the regular grid")
    lon_min, lon_max, lat_min, lat_max = np.min(lon), np.max(lon), np.min(lat), np.max(lat)
    lon_step, lat_step = resolution, resolution
    new_lons = np.arange(lon_min, lon_max, lon_step)
    new_lats = np.arange(lat_min, lat_max, lat_step)
    lon_reg, lat_reg = np.meshgrid(new_lons, new_lats)
    grid_values = np.full(lon_reg.shape, np.nan)

    # initialize an empty merge list
    merge_list = []
        
    # build the mask
    bool_mask = np.isnan(mask)                
    
    # iterate over time
    for t in range(len(dataset.variables['time'])):
        
        # full_data = []
        full_data = np.full((1, len(dataset.dimensions['siglay']), grid_values.shape[0], grid_values.shape[1]), np.nan)

        # debug print
        print("[gen_4dvar] === Processing time %s" % t)
        
        # iterate over depth
        for d in range(len(dataset.dimensions['siglay'])):

            # debug print
            print("[gen_4dvar] === Processing siglay %s" % d)
            
            # map input data on the grid
            points = np.column_stack((lon.flatten(), lat.flatten()))
            values = dataset.variables[varname][t, d, :]
            
            # interpolate
            grid_values = griddata(points, values, (lon_reg, lat_reg), method='linear')

            # mask
            grid_values[bool_mask] = np.nan
            
            # add this layer data to the full array
            full_data[0,d,:,:] = grid_values

        # determine output filename
        output_path, output_basename = os.path.split(output_filename)
        filename = os.path.join(output_path, f"{t:02d}_{varname}.nc")

        # output to NetCDF
        print("[gen_4dvar] === Generating NetCDF file %s" % filename)        
        ds = xr.Dataset(
            {
                varname: (('time', 'depth', 'lat', 'lon'), full_data)
            },
            coords={
                'lon': ('lon', new_lons),
                'lat': ('lat', new_lats),
                'time': ('time', [dataset.variables['time'][0]]),
                'depth': ('depth', range(len(dataset.dimensions['siglay'])))
            }
        )
    
        # Save the dataset
        ds.to_netcdf(filename)
        merge_list.append(filename)
        print("[gen_4dvar] === File %s ready!" % filename)

    # return
    return merge_list



######################################################
#
# Function gen_3dvar
#
######################################################

def gen_3dvar(dataset, resolution, filename, varname, mask):

    # Extract data
    print("[gen_3dvar] === Reading lat and lon")
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    
    # Define the regular grid
    print("[gen_3dvar] === Creating the regular grid")
    lon_min, lon_max, lat_min, lat_max = np.min(lon), np.max(lon), np.min(lat), np.max(lat)
    lon_step, lat_step = resolution, resolution
    new_lons = np.arange(lon_min, lon_max, lon_step)
    new_lats = np.arange(lat_min, lat_max, lat_step)
    lon_reg, lat_reg = np.meshgrid(new_lons, new_lats)
    grid_values = np.full(lon_reg.shape, np.nan)

    # initialize an empty merge list
    merge_list = []
        
    # build the mask
    bool_mask = np.isnan(mask)                

    # init full_data matrix
    full_data = np.full((len(dataset.variables['time']), grid_values.shape[0], grid_values.shape[1]), np.nan)
    
    # iterate over time
    for t in range(len(dataset.variables['time'])):

        # debug print
        print("[gen_3dvar] === Processing time %s" % t)

        # map input data on the grid
        points = np.column_stack((lon.flatten(), lat.flatten()))
        values = dataset.variables[varname][t, :]
            
        # interpolate
        grid_values = griddata(points, values, (lon_reg, lat_reg), method='linear')
        
        # mask
        grid_values[bool_mask] = np.nan
        
        # add this layer data to the full array
        full_data[t,:,:] = grid_values

        # determine output filename
        output_path, output_basename = os.path.split(output_filename)
        filename = os.path.join(output_path, f"{t:02d}_{varname}.nc")

    # output to NetCDF
    print("[gen_3dvar] === Generating NetCDF file %s" % filename)        
    ds = xr.Dataset(
        {
            varname: (('time', 'lat', 'lon'), full_data)
        },
        coords={
            'lon': ('lon', new_lons),
            'lat': ('lat', new_lats),
            'time': ('time', dataset.variables['time'])
        }
    )
    
    # Save the dataset
    ds.to_netcdf(filename)
    merge_list.append(filename)
    print("[gen_3dvar] === File %s ready!" % filename)
    
    # return
    return merge_list



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
    print("[gen_landsea_mask] === Iterating over nv items...")

    nv = dataset.variables['nv'][:].T - 1
    triangles = []
    for i in range(nv.shape[0]):
        
        # Get the vertices of the current triangle and convert them to shapely Polygon
        vertices = [(lon[nv[i, j]], lat[nv[i, j]]) for j in range(3)]
        triangles.append(Polygon(vertices))
        
    # Create a GeoDataFrame to store triangles
    print("[gen_landsea_mask] === Storing triangles in a GeoDataFrame...")    
    gdf_triangles = gpd.GeoDataFrame(geometry=triangles)
        
    # Converts the grid points of the regular grid to Shapely Point objects
    print("[gen_landsea_mask] === Converting grid points in Shapely Points...")
    points = [Point(p) for p in points_regular_flat]

    # Create a GeoDataFrame for the Grid Points (and init to nan)
    print("[gen_landsea_mask] === Storing grid points in a GeoDataFrame...")    
    gdf_points = gpd.GeoDataFrame(geometry=points)
    gdf_points['value'] = np.nan

    # Find grid points belonging to the triangles through the intersection (join) of gdf
    print("[gen_landsea_mask] === Filling the triangles...")    
    inner_points = gpd.sjoin(gdf_points, gdf_triangles, how="inner", predicate='within')
    gdf_points.loc[inner_points.index, 'value'] = 1
   
    
    ######################################################
    #
    # Output to NetCDF File
    #
    ######################################################

    # Prepare data to be stored on NetCDF file
    print("[gen_landsea_mask] === Preparing data for output to NetCDF file...")    
    data = {
        'lat': gdf_points['geometry'].y,
        'lon': gdf_points['geometry'].x,
        'value': gdf_points['value']
    }
    gdf = gpd.GeoDataFrame(data, columns=['lat', 'lon', 'value'])
    
    # convert the pivot dataframe to a pd dataframe
    df = pd.DataFrame(gdf)

    # create a pivot table (useful for conversion)
    pivot_table = df.pivot(index='lat', columns='lon', values='value')

    # convert the pivot table to a numpy array
    land_sea_mask = pivot_table.to_numpy()

    # save original coordinates
    latitudes = pivot_table.index.to_numpy()
    longitudes = pivot_table.columns.to_numpy()
    
    # save to NetCDF
    print("[gen_landsea_mask] === Generating NetCDF file...")    
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
    ds.to_netcdf(mask_filename)
    print("[gen_landsea_mask] === Land sea mask file ready!")

    # return the mask
    return land_sea_mask


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
    # Input file opening and initialization
    #
    ######################################################

    # open dataset
    print("[main] === Opening dataset...")
    input_dataset = nc.Dataset(input_filename)

    # init list of files
    files_to_merge = []    
    
    ######################################################
    #
    # Generation of land sea mask
    #
    ######################################################

    # generate the land sea mask
    print("[main] === Invoking gen_landsea_mask()")
    output_path, output_basename = os.path.split(output_filename)
    mask_filename = "%s/mask_%s" % (output_path, output_basename)
    mask = gen_landsea_mask(input_dataset, resolution_degrees, mask_filename)
    files_to_merge.append(mask_filename)

    ######################################################
    #
    # Generation of bathymetry
    #
    ######################################################

    # generate the land sea mask
    print("[main] === Invoking gen_bathymetry()")
    output_path, output_basename = os.path.split(output_filename)
    bathy_filename = "%s/bathy_%s" % (output_path, output_basename)    
    gen_bathymetry(input_dataset, resolution_degrees, bathy_filename, mask)
    files_to_merge.append(bathy_filename)

    ######################################################
    #
    # Generation of 3d vars
    #
    ######################################################
    
    for v in variables:

        # check if this variable depends on time and node
        if len(input_dataset.variables[v].shape) == 2:

            # debug print
            print("[main] === Processing %s as a 3D var" % v)
            
            # if yes, treat it like a 3d var (node -> lat, lon)
            tmp_merge_list = gen_3dvar(input_dataset, resolution_degrees, output_filename, v, mask)
            for f in tmp_merge_list:
                files_to_merge.append(f)

    
    # ######################################################
    # #
    # # Generation of 4d vars
    # #
    # ######################################################
    
    # for v in variables:

    #     # check if this variable depends on time, node and depth
    #     if len(input_dataset.variables[v].shape) == 3:

    #         # debug print
    #         print("[main] === Processing %s as a 4D var" % v)
            
    #         # if yes, treat it like a 4d var (node -> lat, lon)
    #         tmp_merge_list = gen_4dvar(input_dataset, resolution_degrees, output_filename, v, mask)
    #         for f in tmp_merge_list:
    #             files_to_merge.append(f)


    # ######################################################
    # #
    # # Merge files
    # #
    # ######################################################

    # print("[main] === Will merge:")
    # merge_files(files_to_merge, output_filename)
        
            
    ######################################################
    #
    # End of business
    #
    ######################################################

    # End of business
    print("[main] === EOB. Bye...")    
