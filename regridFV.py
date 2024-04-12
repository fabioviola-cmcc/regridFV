#!/usr/bin/python

######################################################
#
# Requirements
#
######################################################

import os
import pdb
import sys
import dask
import getopt
import traceback
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
from concurrent.futures import ProcessPoolExecutor


# Set up Dask to manage memory usage more effectively
dask.config.set({'array.slicing.split_large_chunks': True})


######################################################
#
# Function open_datasets
#
######################################################

def open_datasets(file_list):

    """
    Handler to deal with very large data
    """
    
    # Use dask's automatic chunking to deal with large data
    return [xr.open_dataset(f, chunks={'time': 1}) for f in file_list]


######################################################
#
# Function merge_datasets
#
######################################################

def merge_datasets(datasets):

    """
    Utility to merge datasets exploting dask
    """
    
    # Use Dask's lazy loading
    return xr.concat(datasets, dim="time")


######################################################
#
# Function process_timestep4d
#
######################################################

def process_timestep4d(t, data, depth_levels, varname, grid_values, lon_reg, lat_reg, bool_mask, interp, output_dir, prefix, nele, coords):

    """
    Function to process a timestep over all the siglay for a
    4d variable. This function is run as a pool of concurrent processes
    """

    # initialize the output data structure
    full_data = np.full((1, depth_levels, grid_values.shape[0], grid_values.shape[1]), np.nan)

    # iterate over the depth layers
    for d in range(depth_levels):
        if nele:
            points = np.column_stack((coords['lonc'].flatten(), coords['latc'].flatten()))
        else:
            points = np.column_stack((coords['lon'].flatten(), coords['lat'].flatten()))
        values = data[d, :]

        # interpolate
        grid_values = griddata(points, values, (lon_reg, lat_reg), method=interp)
        grid_values[bool_mask] = np.nan        
        full_data[0,d,:,:] = grid_values

    # Build the output file
    filename = os.path.join(output_dir, f"{prefix}_{t:02d}_{varname}.nc")   
    ds = xr.Dataset(
        {
            varname: (('time', 'depth', 'lat', 'lon'), full_data)
        },
        coords={
            'lon': ('lon', coords['new_lons']),
            'lat': ('lat', coords['new_lats']),
            'time': ('time', [t]),
            'depth': ('depth', range(depth_levels))
        }
    )
    encoding = {varname: {"dtype": "float32"}}
    ds.to_netcdf(filename, encoding=encoding)
    print(f"[gen_4dvar] === File {filename} ready!")

    # return the filename, to be later used for merging
    return filename

    
######################################################
#
# Function merge_files
#
######################################################

def merge_files(merge_list, filename, avg_filename):

    """
    As the name suggests, this function is used to merge
    multiple NetCDF files into a single dataset
    """

    # open and merge datasets    
    datasets = open_datasets(merge_list)
    ds_merged = merge_datasets(datasets)
    
    # save the new dataset into a single file
    print("[merge_files] === Merged input files to %s" % filename)    
    ds_merged.to_netcdf(filename, engine='h5netcdf', mode='w', format='NETCDF4')

    # remove input files
    for f in merge_list:
        os.remove(f)

    # also calculate the average
    print("[merge_files] === Average saved on %s" % avg_filename)        
    ds_avg = ds_merged.mean(dim='time', keep_attrs=True, skipna=True)    
    ds_avg.to_netcdf(avg_filename, engine='h5netcdf', mode='w', format='NETCDF4')

    
######################################################
#
# Function gen_bathymetry
#
######################################################

def gen_bathymetry(dataset, resolution, filename, mask, interp, bbox):

    """
    Function to generate the bathymetry
    """
    
    # extract lat and lon data
    print("[gen_bathymetry] === Reading lat and lon")
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    
    # define the regular grid
    print("[gen_bathymetry] === Creating the regular grid")
    lat_min, lat_max, lon_min, lon_max = float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3]) #np.min(lon), np.max(lon), np.min(lat), np.max(lat)
    lon_step, lat_step = resolution, resolution

    try:
        new_lons = np.arange(lon_min, lon_max, lon_step)
        new_lats = np.arange(lat_min, lat_max, lat_step)
    except:
        print(traceback.print_exc())
        pdb.set_trace()
        
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

def gen_4dvar(dataset, resolution, output_dir, prefix, varname, mask, nele, interp, bbox):

    """
    Function to regularize a 4d variable
    """
    
    # Extract data
    print("[gen_4dvar] === Reading lat and lon")
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    lonc = dataset.variables['lonc'][:]
    latc = dataset.variables['latc'][:]
    
    # Define the regular grid
    print("[gen_4dvar] === Creating the regular grid")
    lat_min, lat_max, lon_min, lon_max = float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3])
    lon_step, lat_step = resolution, resolution
    new_lons = np.arange(lon_min, lon_max, lon_step)
    new_lats = np.arange(lat_min, lat_max, lat_step)
    lon_reg, lat_reg = np.meshgrid(new_lons, new_lats)
    grid_values = np.full(lon_reg.shape, np.nan)

    # define an exchange data structure
    coords = {'lon': lon, 'lat': lat, 'lonc': lonc, 'latc': latc, 'new_lons': new_lons, 'new_lats': new_lats }
    
    # initialize an empty merge list
    merge_list = []
        
    # build the mask
    bool_mask = np.isnan(mask)

    # start a pool of processes to simultaneously produce files
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_timestep4d, t, dataset.variables[varname][t, :, :], len(dataset.dimensions['siglay']), varname, grid_values, lon_reg, lat_reg, bool_mask, interp, output_dir, prefix, nele, coords)
                   for t in range(len(dataset.variables['time']))]
        merge_list = [f.result() for f in futures]

    # merge files and remove single ones
    filename = os.path.join(output_dir, f"{prefix}_{varname}.nc")
    avg_filename = os.path.join(output_dir, f"{prefix}_{varname}_AVG.nc")        
    merge_files(merge_list, filename, avg_filename)

            
######################################################
#
# Function gen_3dvar
#
######################################################

def gen_3dvar(dataset, resolution, output_dir, prefix, varname, mask, interp, bbox):

    """
    Function to regularize a 3D variable
    """
    
    # Extract data
    print("[gen_3dvar] === Reading lat and lon")
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    
    # Define the regular grid
    print("[gen_3dvar] === Creating the regular grid")
    lat_min, lat_max, lon_min, lon_max = float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3])
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

    # determine output filenames
    filename = os.path.join(output_dir, f"{prefix}_{varname}.nc")
    avg_filename = os.path.join(output_dir, f"{prefix}_{varname}_AVG.nc")    

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

    encoding = {
        varname: { "dtype" : "float32" }
    } 
           
    # Save the dataset
    ds.to_netcdf(filename, encoding=encoding)
    print("[gen_3dvar] === File %s ready!" % filename)

    # average on time
    print("[merge_files] === Average saved on %s" % avg_filename)        
    ds_avg = ds.mean(dim='time', keep_attrs=True, skipna=True)    
    ds_avg.to_netcdf(avg_filename, engine='h5netcdf', mode='w', format='NETCDF4')
    

######################################################
#
# Function gen_landsea_mask
#
######################################################

def gen_landsea_mask(dataset, resolution, filename, bbox):

    """
    Function to generate the landsea mask
    """
    
    # ===== Define the regular grid and build land-sea mask =====

    # Extract data
    print("[gen_landsea_mask] === Reading lat and lon")
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    
    # Define the regular grid
    print("[gen_landsea_mask] === Creating the regular grid")
    lat_min, lat_max, lon_min, lon_max = float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3]) #np.min(lon), np.max(lon), np.min(lat), np.max(lat)
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
    options, remainder = getopt.getopt(sys.argv[1:], 'i:o:r:v:p:m:b:', ['input=', 'output=', 'resolution=', 'variables=', 'prefix=', 'interp=', 'bbox='])

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

        elif opt in ('-b', '--bbox'):
            bbox = arg.split(":")
            print("[main] === Bounding box set to:")
            print("[main] === - Min lat: %s" % bbox[0])
            print("[main] === - Min lon: %s" % bbox[1])            
            print("[main] === - Max lat: %s" % bbox[2])
            print("[main] === - Max lon: %s" % bbox[3])            

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
    mask = gen_landsea_mask(input_dataset, resolution_degrees, mask_filename, bbox)

    # ===== Generation of bathymetry =====

    # generate the land sea mask
    print("[main] === Invoking gen_bathymetry()")
    bathy_filename = os.path.join(output_directory, "%s_bathymetry.nc" % (prefix))
    gen_bathymetry(input_dataset, resolution_degrees, bathy_filename, mask, interp, bbox)

    # ===== Generation of 3d vars =====
    
    for v in variables:

        # check if this variable depends on time and node
        if len(input_dataset.variables[v].shape) == 2:
            
            # if yes, treat it like a 3d var (node -> lat, lon)
            print("[main] === Processing %s as a 3D var" % v)
            gen_3dvar(input_dataset, resolution_degrees, output_directory, prefix, v, mask, interp, bbox)
            
    # ===== Generation of 4d vars =====
    
    for v in variables:

        # check the number of dimensions
        if len(input_dataset.variables[v].shape) == 3:

            # check if this variable depends on time, node and depth
            if "nele" not in input_dataset.variables[v].dimensions:
                        
                # if yes, treat it like a 4d var (node -> lat, lon)
                print("[main] === Processing %s as a 4D time/depth/lat/lon var" % v)                
                gen_4dvar(input_dataset, resolution_degrees, output_directory, prefix, v, mask, False, interp, bbox)

            else:
            
                # if yes, treat it like a 4d var (node -> lat, lon)
                print("[main] === Processing %s as a 4D time/depth/lat/lon nele-based var" % v)                
                gen_4dvar(input_dataset, resolution_degrees, output_directory, prefix, v, mask, True, interp, bbox)
                                
    # End of business
    print("[main] === EOB. Bye...")    
