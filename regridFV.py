#!/usr/bin/python

######################################################
#
# Requirements
#
######################################################

import os
import sys
import getopt
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import geopandas as gpd
from matplotlib.path import Path
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, Point
from scipy.interpolate import LinearNDInterpolator


######################################################
#
# Function merge_files
#
######################################################

def merge_files(merge_list, filename):

    """
    As the name suggests, this function is used to merge
    multiple NetCDF files into a single dataset
    """

    # open the datasets
    datasets = []
    for m in merge_list:
        datasets.append(xr.open_dataset(m))

    # merge the datasets
    ds_merged = xr.merge(datasets, compat='override')
    
    # save the new dataset into a single file
    print("[merge_files] === Merged input files to %s" % filename)    
    ds_merged.to_netcdf(filename)

    # remove input files
    for f in merge_list:
        os.remove(f)
    
    
######################################################
#
# Function gen_bathymetry
#
######################################################

def gen_bathymetry(dataset, resolution, filename, mask, interp):

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
    grid_values = griddata(points, values, (lon_reg, lat_reg), method=interp) 
        
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

def gen_4dvar(dataset, resolution, output_dir, prefix, varname, mask, nele, interp):

    # Extract data
    print("[gen_4dvar] === Reading lat and lon")
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    lonc = dataset.variables['lonc'][:]
    latc = dataset.variables['latc'][:]
    
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
        
        # iterate over depth
        for d in range(len(dataset.dimensions['siglay'])):
            
            # map input data on the grid
            if nele:
                points = np.column_stack((lonc.flatten(), latc.flatten()))
            else:
                points = np.column_stack((lon.flatten(), lat.flatten()))
            values = dataset.variables[varname][t, d, :]
            
            # interpolate
            grid_values = griddata(points, values, (lon_reg, lat_reg), method=interp)

            # mask
            grid_values[bool_mask] = np.nan
            
            # add this layer data to the full array
            full_data[0,d,:,:] = grid_values

        # determine output filename
        filename = os.path.join(output_dir, f"{prefix}_{t:02d}_{varname}.nc")

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


    # merge files and remove single ones
    filename = os.path.join(output_dir, f"{prefix}_{varname}_{t}.nc")    
    merge_files(merge_list, filename)

            
######################################################
#
# Function gen_3dvar
#
######################################################

def gen_3dvar(dataset, resolution, output_dir, prefix, varname, mask, interp):

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

        # map input data on the grid
        points = np.column_stack((lon.flatten(), lat.flatten()))
        values = dataset.variables[varname][t, :]
            
        # interpolate
        grid_values = griddata(points, values, (lon_reg, lat_reg), method=interp)
        
        # mask
        grid_values[bool_mask] = np.nan
        
        # add this layer data to the full array
        full_data[t,:,:] = grid_values

    # determine output filename
    filename = os.path.join(output_dir, f"{prefix}_{varname}.nc")

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
    print("[gen_3dvar] === File %s ready!" % filename)


######################################################
#
# Function gen_landsea_mask
#
######################################################

def gen_landsea_mask(dataset, resolution, filename):

    # ===== Define the regular grid and build land-sea mask =====

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

    # ===== Fill triangles in land sea mask =====

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
   
    # ===== Output to NetCDF File =====

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
    ds.to_netcdf(filename)
    print("[gen_landsea_mask] === Land sea mask file ready!")

    # return the mask
    return land_sea_mask


######################################################
#
# Main
#
######################################################

if __name__ == "__main__":

    # ===== Input params management =====
    
    # read input params
    options, remainder = getopt.getopt(sys.argv[1:], 'i:o:r:v:p:m:', ['input=', 'output=', 'resolution=', 'variables=', 'prefix=', 'interp='])

    # parse input params
    for opt, arg in options:
        
        if opt in ('-o', '--output'):
            output_directory = arg
            print("[main] === Output file set to: %s" % output_directory)

        elif opt in ('-i', '--input'):
            input_filename = arg
            print("[main] === Input file set to: %s" % input_filename)

        elif opt in ('-p', '--prefix'):
            prefix = arg
            print("[main] === Prefix for output file set to: %s" % prefix)

        elif opt in ('-m', '--interp'):
            interp = arg
            print("[main] === Interpolation method set to: %s" % interp)

        elif opt in ('-r', '--resolution'):
            resolution_meters = arg
            resolution_degrees = round(float(resolution_meters) / 111319, 6)
            print("[main] === Output resolution set to: %s m (%s deg E)" % (resolution_meters, round(resolution_degrees, 6)))

        elif opt in ('-v', '--variables'):
            variables = arg.split(":")
            print("[main] === Variables of interest are: %s" % " ".join(variables))            
            
        else:
            print("[main] === Unrecognized option %s. Will be ignored." % opt)

    # ===== Input file opening and initialization =====

    # open dataset
    print("[main] === Opening dataset...")
    input_dataset = nc.Dataset(input_filename)

    # init list of files
    files_to_merge = []    

    # ===== Generation of land sea mask =====

    # generate the land sea mask
    print("[main] === Invoking gen_landsea_mask()")
    mask_filename = "%s/%s_landSeaMask.nc" % (output_directory, prefix)
    mask = gen_landsea_mask(input_dataset, resolution_degrees, mask_filename)

    # ===== Generation of bathymetry =====

    # generate the land sea mask
    print("[main] === Invoking gen_bathymetry()")
    bathy_filename = os.path.join(output_directory, "%s_bathymetry.nc" % (prefix))
    gen_bathymetry(input_dataset, resolution_degrees, bathy_filename, mask, interp)
    
    # ===== Generation of 3d vars =====
    
    for v in variables:

        # check if this variable depends on time and node
        if len(input_dataset.variables[v].shape) == 2:
            
            # if yes, treat it like a 3d var (node -> lat, lon)
            print("[main] === Processing %s as a 3D var" % v)
            gen_3dvar(input_dataset, resolution_degrees, output_directory, prefix, v, mask, interp)
            
    # ===== Generation of 4d vars =====
    
    for v in variables:

        # check the number of dimensions
        if len(input_dataset.variables[v].shape) == 3:

            # check if this variable depends on time, node and depth
            if "nele" not in input_dataset.variables[v].dimensions:
                        
                # if yes, treat it like a 4d var (node -> lat, lon)
                print("[main] === Processing %s as a 4D time/depth/lat/lon var" % v)                
                gen_4dvar(input_dataset, resolution_degrees, output_directory, prefix, v, mask, False, interp)

            else:
            
                # if yes, treat it like a 4d var (node -> lat, lon)
                print("[main] === Processing %s as a 4D time/depth/lat/lon nele-based var" % v)                
                gen_4dvar(input_dataset, resolution_degrees, output_directory, prefix, v, mask, True, interp)
                                
    # End of business
    print("[main] === EOB. Bye...")    