# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:00:49 2023

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
    splevelfolderdir=root+os.sep+r"result\relation\table\fish_div_LULC_effect_species_level"
    catchfolderdir=root+os.sep+r"data\RS\catch_data\catchmask_site"            #catchment map folder
    resultfolderdir=root+os.sep+r"result\relation\table\river_biodiv_species_distrib" #result folder
    cufile=root+os.sep+r"code\src\calcRiverBiodiv_SP_HS.cu"                    #cu file for catchment calculation with CUDA   
    
    decay_func=1    #1: exp, 2: sph, 3: matern15, 4: matern25, 5: gaussian  
    decay_func_str="exp"
    FA_thres=100000         #unit: num. of pixels in catchment (for major river channels) using flow accumulation map
    radius_pix=np.int32(5/0.09)                                               #searching radius in pixel (km/length of pixel)
    env_var_name="discharge"                                                   #environmental variable to fit the baseline estimation
    
    lower_dist=2                                                               #we search for the optimal species-level model within a certain distance boundary 
    upper_dist=100                                                             #please indicate the upper and lower boundaries 
    
    #read species list file
    splistfilename="fish_div_spnames.csv"
    splistfiledir=biodivfolderdir+os.sep+splistfilename
    splist=list(init.readCSVasPandas(splistfiledir)["spname"])                 #read the data and 
    num_sp=len(splist)
    
    #flow direction raster file    
    FD_filename="THA_HydroSHEDS_90_FlowDirec.tif"      
    FD_filedir=hydshfolderdir+os.sep+FD_filename
    
    #flow accumulation raster file
    FA_filename="THA_HydroSHEDS_90_FlowAccum.tif"
    FA_filedir=hydshfolderdir+os.sep+FA_filename
    
    #river discharge raster file
    Q_filename="THA_HydroRIVERS_90_Q.tif"
    Q_filedir=hydshfolderdir+os.sep+Q_filename

#%%    
    #read the flow direction map 
    [mapLayers,driver,geoTrans_FD,proj,nrow,ncol]=ptf.readTiffAsNumpy([FD_filedir],datatype='UInt8')
    FD=mapLayers[:,:,0]
    del mapLayers
    
    #read the flow accumulation map 
    [mapLayers,driver,geoTrans_FD,proj,nrow,ncol]=ptf.readTiffAsNumpy([FA_filedir],datatype='Int32')
    FA=mapLayers[:,:,0]
    del mapLayers    
    
    #read the river discharge map 
    [mapLayers,driver,geoTrans_FD,proj,nrow,ncol]=ptf.readTiffAsNumpy([Q_filedir],datatype='Float32')
    Q=mapLayers[:,:,0]
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
    
    [DX,DY]=catch.calcPixelDXDYArray(nrow,ncol,geoTrans_FD)                    #calculate pixel length DX DY 
                                                                               #using the haversine formula (this function globally applicable)        
  
#%%
    #collect the optimal species-level models 
    #check and collect species list
    splist_exists=[]
    splevelFileList=[]
    for i_sp in range(num_sp):
        splevelfilename="LULC_effect_"+decay_func_str+"_var_"+env_var_name+"_SP"+str(i_sp+1).zfill(3)+"_"+splist[i_sp]+".csv"
        splevelfiledir=splevelfolderdir+os.sep+splevelfilename
        if os.path.exists(splevelfiledir):
            splist_exists.append("SP"+str(i_sp+1).zfill(3))
            splevelFileList.append(splevelfiledir)
    num_sp=len(splist_exists)
    #find optimal values for all the BS simlulations
    num_sp_mat_cols=num_LULC_types+6
    SP_MAT=np.matrix(np.zeros([num_sp,num_sp_mat_cols],dtype=np.float32))    
    for i_sp in range(num_sp):
        splevelfiledir=splevelFileList[i_sp]
        param_data=np.array(init.readCSVasPandas(splevelfiledir))[:,1:num_sp_mat_cols+1].astype(np.float32)
        param_dists=param_data[:,num_LULC_types+2]
        bool_dist=(param_dists>=lower_dist)&(param_dists<upper_dist)
        param_data=param_data[bool_dist,:]
        #find the ID with highest TSS value but lowest abs(thres-0.5)
        param_eval_max=np.max(param_data[:,num_LULC_types+4])
        param_eval_max_ids=np.argwhere(param_data[:,num_LULC_types+4]==param_eval_max).ravel()#[::-1]        
        param_max_id=param_eval_max_ids[np.argmin(np.abs(0.5-param_data[param_eval_max_ids,num_LULC_types+3]))]
        # param_max_id=param_eval_max_ids[np.argmin(np.abs(0.15-param_data[param_eval_max_ids,num_LC_types+2]))]
        
        SP_MAT[i_sp,0:num_LULC_types]=param_data[param_max_id,1:(num_LULC_types+1)]
        SP_MAT[i_sp,num_LULC_types]=param_data[param_max_id,0]  
        SP_MAT[i_sp,num_LULC_types+1]=param_data[param_max_id,num_LULC_types+1]
        SP_MAT[i_sp,(num_LULC_types+2):(num_sp_mat_cols)]=param_data[param_max_id,(num_LULC_types+2):(num_sp_mat_cols)]

#%%
    #extract river channel pixels
    river_pixels=catch.extRiverPixels(FA,FA_thres)
    num_rivpix=river_pixels.shape[0]
    rivPixEnv=np.array(np.concatenate([river_pixels,Q[river_pixels[:,0],river_pixels[:,1]]],axis=1,dtype=np.float32))
    rivPixEnv[:,2]=np.log(rivPixEnv[:,2]+1e-3)                                  #log-transfer the river discharge to match the model

    #prepare the result array (probability prediction from the model)
    B=np.zeros([num_rivpix,num_sp],dtype=np.float32)
        
    #CUDA implementation
    #flow directions & max loop should be specified in CUDA file. 
    #step 1: read cu file and compile the CUDA code
    with open(cufile,"rt") as f:
        cu_src = f.read()
    
    mod = SourceModule(cu_src)
    #step 2: specify the block & grid sizes in CUDA kernel
    BLOCKDIM=256
    blockSize=(BLOCKDIM,1,1)
    bx=int((num_rivpix+BLOCKDIM-1)/BLOCKDIM)
    gridSize=(bx,1,1)

    #step 3: link kernel function and run the calculation process
    #nrow, ncol etc. need to be transferred to Int32 type to fit CUDA kernel
    t1=time.time()
    func = mod.get_function("calcRiverBiodivSP")
    func(cuda.InOut(B),cuda.In(rivPixEnv),cuda.In(FD),cuda.In(ras_LULC),cuda.In(DX),cuda.In(DY),cuda.In(SP_MAT),cuda.In(LULC_types),\
         np.int32(decay_func),np.int32(num_LULC_types),np.int32(num_sp_mat_cols),np.int32(radius_pix),\
         np.int32(num_rivpix),np.int32(num_sp),np.int32(nrow),np.int32(ncol),block=blockSize,grid=gridSize)
    cuda.Context.synchronize()
    t2=time.time()
    t2t1_hour=(t2-t1)/3600
    print("GPU computing finished.    time = %.2f h.  "%t2t1_hour)
#%%
    #predict presence / absence of species distribution
    B=1/(1+np.exp(-1*B))
    for i_sp in range(num_sp):
        B_sp=B[:,i_sp]
        B_sp[B_sp>SP_MAT[i_sp,num_LULC_types+3]]=1
        B_sp[B_sp<=SP_MAT[i_sp,num_LULC_types+3]]=0
        B[:,i_sp]=B_sp
#%%
    #write results
    resultData=np.array(np.concatenate([river_pixels,B],axis=1,dtype=np.float32))
    if not os.path.exists(resultfolderdir):
        os.makedirs(resultfolderdir)
    csvfilename1="river_species_distrib_map.csv"
    csvfiledir1=resultfolderdir+os.sep+csvfilename1
    init.writeArrayToCSV(resultData,["Y","X"]+splist_exists,csvfiledir1)
    print("ALL DONE.\n")    
    