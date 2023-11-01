# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 13:33:46 2023

@author: Heng Zhang (heng.zhang@eawag.ch; hengzhang.zhh@gmail.com) at Prof. Florian Altermatt Lab (https://www.altermattlab.ch/)
@Swiss Federal Institute of Aquatic Science and Technology (EAWAG / ETH) & University of Zurich (UZH). 
@Please cite the paper below when using any part of this code/project. 

Heng Zhang, et al., Terrestrial land cover shapes fish diversity across a major subtropical basin. 
BioRxiv link: 
    
"""

import os
import time
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
    root=r"D:\Programs\eDNA_RS_Landscape\github"                               #root directory of the project
    LULCfolderdir=root+os.sep+r"data\RS\land_cover"                            #land use/land cover map folder
    hydshfolderdir=root+os.sep+r"data\RS\catch_data\HydroSHEDS_V1"             #HydroSHEDS data in this case
    biodivfolderdir=root+os.sep+r"data\eDNA"                                   #csv folder to store eDNA-biodiversity data
    catchfolderdir=root+os.sep+r"data\RS\catch_data\catchmask_site"            #catchment map folder
    resultfolderdir=root+os.sep+r"result\relation\map\terrestrial_LULC_effect" #result folder
    cufile=root+os.sep+r"code\src\calcAccumLULCEffect.cu"                      #cu file for catchment calculation with CUDA    

    decay_func=1    #1: exp, 2: sph, 3: matern15, 4: matern25, 5: gaussian  
    decay_func_str="exp"
    #please indicate the optimal parameter values here.
    dist=19
    LULC_effect_values=[1.438477,-0.23793,-2.16344,-4.85732,-3.68412]    
    unit_area=1             #unit: km2
    FA_thres=100000         #unit: num. of pixels in catchment (for major river channels) using flow accumulation map
    max_dist=dist
    
    #flow direction raster file    
    FD_filename="THA_HydroSHEDS_90_FlowDirec.tif"      
    FD_filedir=hydshfolderdir+os.sep+FD_filename
    
    #flow accumulation raster file
    FA_filename="THA_HydroSHEDS_90_FlowAccum.tif"
    FA_filedir=hydshfolderdir+os.sep+FA_filename
#%%    
    #read the flow direction map layer
    [mapLayers,driver,geoTrans_FD,proj,nrow,ncol]=ptf.readTiffAsNumpy([FD_filedir],datatype='UInt8')
    FD=mapLayers[:,:,0]
    del mapLayers
    
    #read the flow accumulation map layer
    [mapLayers,driver,geoTrans_FD,proj,nrow,ncol]=ptf.readTiffAsNumpy([FA_filedir],datatype='Int32')
    FA=mapLayers[:,:,0]
    del mapLayers
    
    #read LULC map
    LULCfilename="ESACCI_LC_L4_LCCS_2016_THA_RecBound.tif"
    LULCfiledir=LULCfolderdir+os.sep+LULCfilename
    [ras_LULC,driver_LULC,geoTrans_LULC,proj_LULC,nrow_LULC,ncol_LULC]=ptf.readTiffAsNumpy([LULCfiledir],datatype='UInt8')
    ras_LULC=ras_LULC[:,:,0]
    
    #read land cover map classification system
    cvtfilename="LULC_cvt_code.csv"
    LULC_CVT=init.readCSVasPandas(LULCfolderdir+os.sep+cvtfilename)
    LULC_types=np.array(LULC_CVT["CUSTOM_VALUE"].unique(),dtype=np.int32)
    LULC_types=LULC_types[LULC_types!=0]
    ras_LULC=catch.cvtLCTypes(ras_LULC,LULC_CVT,"ESACCI_VALUE","CUSTOM_VALUE")  
    ras_LULC=ptf.resizeImage(ras_LULC,[ncol,nrow])
    ras_LULC=np.uint8(ras_LULC)
    num_LULC_types=len(LULC_types)
        
    LULC_types=np.int32(LULC_types).reshape([num_LULC_types,1])
    param_vec=np.float32(np.repeat(dist,num_LULC_types)).reshape([num_LULC_types,1])  #vector of optimal distance (same value in this case)
    LULC_effect_values=np.float32(LULC_effect_values).reshape([num_LULC_types,1])
    
    [DX,DY]=catch.calcPixelDXDYArray(nrow,ncol,geoTrans_FD)                    #calculate pixel length DX DY 
                                                                               #using the haversine formula (this function globally applicable)        
    LULC_Effect=np.zeros([nrow,ncol],dtype=np.float32)
    
#%%    
    #CUDA implementation
    #flow directions & max loop should be specified in CUDA file. 
    #step 1: read cu file and compile the CUDA code
    with open(cufile,"rt") as f:
        cu_src = f.read()
    
    mod = SourceModule(cu_src)
    #step 2: specify the block & grid sizes in CUDA kernel
    BLOCKDIM=16
    blockSize=(BLOCKDIM,BLOCKDIM,1)
    bx=int((ncol+BLOCKDIM-1)/BLOCKDIM)
    by=int((nrow+BLOCKDIM-1)/BLOCKDIM)
    gridSize=(bx,by,1)

    #step 3: link kernel function and run the calculation process
    #nrow, ncol etc. need to be transferred to Int32 type to fit CUDA kernel
    t1=time.time()
    func = mod.get_function("calcAccumLULCEffect")
    func(cuda.InOut(LULC_Effect),cuda.In(FA),cuda.In(FD),cuda.In(ras_LULC),cuda.In(DX),cuda.In(DY),cuda.In(param_vec),cuda.In(LULC_effect_values),\
         cuda.In(LULC_types),np.int32(decay_func),np.int32(num_LULC_types),np.float32(FA_thres),\
         np.float32(max_dist),np.float32(unit_area),np.int32(nrow),np.int32(ncol),block=blockSize,grid=gridSize)
    cuda.Context.synchronize()
    
    t2=time.time()
    t2t1_s=(t2-t1)
    print("GPU computing finished.    time = %.2f s.  "%t2t1_s)    
#%%   
    #write the results
    if not os.path.exists(resultfolderdir):
        os.makedirs(resultfolderdir)
    resultfilename="terrestrial_LULC_effect_map.tif"
    resultfiledir=resultfolderdir+os.sep+resultfilename
    ptf.writeNumpyToTiff(LULC_Effect,driver,geoTrans_FD,proj,nrow,ncol,-9999,resultfiledir,datatype='Float32')
    print("ALL DONE.")