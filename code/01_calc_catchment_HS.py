# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:35:00 2023

@author: Heng Zhang (heng.zhang@eawag.ch; hengzhang.zhh@gmail.com) at Prof. Florian Altermatt Lab (https://www.altermattlab.ch/)
@Swiss Federal Institute of Aquatic Science and Technology (EAWAG / ETH Domain) & University of Zurich (UZH). 
@Please cite the paper below when using any part of this code/project. 

Heng Zhang, et al., Terrestrial land cover shapes fish diversity across a major subtropical basin. 
BioRxiv link: https://doi.org/10.1101/2023.10.30.564688

"""

#required packages: gdal & pycuda
#please note that the catchment computing only runs on a NVIDIA GPU. 
#please translate the CUDA kernal code to OpenCL/C++ if you would like to use an AMD GPU or even CPUs

import os
import time
import copy
import numpy as np

#CUDA environment
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

#local functions
import src.init as init
import src.catch_funcs as catch
import src.processgeotiff4 as ptf

#%%
if __name__=="__main__":
    #please specify the local file directory
    root=r"D:\Programs\eDNA_RS_Landscape\github"                               #root directory of the project
    hydshfolderdir=root+os.sep+r"data\RS\catch_data\HydroSHEDS_V1"             #HydroSHEDS data in this case
    biodivfolderdir=root+os.sep+r"data\eDNA"                                   #csv folder to store eDNA-biodiversity data
    resultdirto=root+os.sep+r"data\RS\catch_data\catchmask_site"               #result folder
    cufile=root+os.sep+r"code\src\calcFlowDist_HS.cu"                          #cu file for catchment calculation with CUDA
    
    #read biodiversity data with coordinates
    biodivfilename="fish_div_coords.csv"
    biodivfiledir=biodivfolderdir+os.sep+biodivfilename
    biodivPD=init.readCSVasPandas(biodivfiledir)
    
    #read flow direction (FD) raster 
    FD_filename="THA_HydroSHEDS_90_FlowDirec.tif"      
    FD_filedir=hydshfolderdir+os.sep+FD_filename

    #read the flow direction map layer: please convert the FD raster to np.uint8 format
    [FD,driver,geoTrans_FD,proj,nrow_FD,ncol_FD]=ptf.readTiffAsNumpy([FD_filedir],datatype='UInt8')
    FD=FD[:,:,0]

    radius_pix=1                                                               #radius in pixel for flow outlets (sampling sites)
    default_dist_value=geoTrans_FD[1]*111*radius_pix                           #default flow distance value for the outlet pixels
    max_trace_steps=int(10000/0.09)                                            #maximal tracing steps in the catchment computation (see cufile)

    bool_clip_result=True
    max_clip_dist=600                                                          #unit: km
    
    #!!!please calibrate sampling sites to the major river channels in HydroSHEDS before computing catchment. 
    lon_header="lon_HydroSHEDS"                                                #header for longitudes of calibrated sites
    lat_header="lat_HydroSHEDS"                                                #header for latitudes of calibrated sites

    siteIDList=list(biodivPD["SiteID"])
    num_sites=len(siteIDList)

    #CUDA implementation: read cu file and compile the CUDA kernel code
    #please set the correct flow direction vector (var: dirvecs) in the cu file.
    with open(cufile,"rt") as f:
        cu_src = f.read()
    
    mod = SourceModule(cu_src)
    func = mod.get_function("calcFlowDist")
    #%%
    #iteration: calculate catchment for each site
    for i_site in range(num_sites):
        siteID=siteIDList[i_site]
        print("computing catchment...        site no. (%d / %d)"%((i_site+1),num_sites))    
        #update the geoinfo of catchment map (the geoinfo may change in the clipping step below)
        [nrow,ncol]=[nrow_FD,ncol_FD]       
        geoTrans=np.float32(copy.deepcopy(geoTrans_FD))
        site_coord=[biodivPD.loc[i_site,lon_header],biodivPD.loc[i_site,lat_header]]        #get the coordinate of sampling site
        dst_points=catch.genDstPoints(site_coord,geoTrans,radius_dist=geoTrans[1]*radius_pix)
        if len(dst_points)==0 or np.min(dst_points)<0:
            print("site out of catchment boundary! computation for next site continues.\n")
            continue
        
        t1=time.time()
        catchdist=catch.initCatchDist(dst_points,nrow,ncol,default_dist_value)
        [DX,DY]=catch.calcPixelDXDYArray(nrow,ncol,geoTrans)                   #calculate pixel length DX DY 
                                                                               #using the haversine formula (this function globally applicable)        
        #%%    
        #specify the block & grid sizes in CUDA computing
        BLOCKDIM=16
        blockSize=(BLOCKDIM,BLOCKDIM,1)
        bx=int((ncol+BLOCKDIM-1)/BLOCKDIM)
        by=int((nrow+BLOCKDIM-1)/BLOCKDIM)
        gridSize=(bx,by,1)
        
        #calculate catchment for sampling site on NVIDIA GPU
        #max_trace_steps, nrow, ncol, etc. need to be transferred to Int32 type to match CUDA kernel functions
        #please check "__global__ void calcFlowDist()" function for corresponding variable types.
        func(cuda.InOut(catchdist),cuda.In(FD),cuda.In(DX),cuda.In(DY),\
             np.float32(default_dist_value),np.int32(max_trace_steps),np.int32(nrow),np.int32(ncol),\
                 block=blockSize,grid=gridSize)
        cuda.Context.synchronize()                
        #%%
        #clip catchment map with flow distance with max_clip_dist
        if bool_clip_result:
            catchdist[catchdist>max_clip_dist]=0
            [catchdist,geoTrans,nrow,ncol]=ptf.clipRas2Rect(catchdist,geoTrans)
        t2=time.time()
        
        #write catchment map to local directory
        catchdistfilename="site_ID_"+str(siteID).zfill(2)+"_catchdist.tif"
        if not os.path.exists(resultdirto):
            os.makedirs(resultdirto)
        catchdistfiledir=resultdirto+os.sep+catchdistfilename            
        [driver,proj]=ptf.createTIFFDriverFromEPSG(4326)                       #generate the driver of TIFF raster and the projection of WGS_1984 (lon-lat format)
        ptf.writeNumpyToTiff(catchdist,driver,geoTrans,proj,nrow,ncol,-9999,catchdistfiledir,datatype='Float32')
        print("computation finished!      time = %.2f s \n"%(t2-t1))        
    print("ALL DONE.\n")
