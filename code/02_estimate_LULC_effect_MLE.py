# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:21:35 2023

@author: Heng Zhang (heng.zhang@eawag.ch; hengzhang.zhh@gmail.com) at Prof. Florian Altermatt Lab (https://www.altermattlab.ch/)
@Swiss Federal Institute of Aquatic Science and Technology (EAWAG / ETH Domain) & University of Zurich (UZH). 
@Please cite the paper below when using any part of this code/project. 

Heng Zhang, et al., Terrestrial land cover shapes fish diversity across a major subtropical basin. 
BioRxiv link: https://doi.org/10.1101/2023.10.30.564688
    
"""

import os
import numpy as np

import src.init as init
import src.catch_funcs as catch
import src.processgeotiff4 as ptf
import src.model_funcs_FishDiv_LULC as model

#%%
if __name__=="__main__":
    root=r"D:\Programs\eDNA_RS_Landscape\github"                               #root directory of the project
    LULCfolderdir=root+os.sep+r"data\RS\land_cover"
    biodivfolderdir=root+os.sep+r"data\eDNA"                                   #csv folder to store eDNA-biodiversity data
    catchfolderdir=root+os.sep+r"data\RS\catch_data\catchmask_site"            #catchment map folder
    resultfolderdir=root+os.sep+r"result\relation\table\fish_div_LULC_effect"  #result folder
    
    #read biodiversity data with coordinates
    biodivfilename="fish_div_coords.csv"
    biodivfiledir=biodivfolderdir+os.sep+biodivfilename
    biodivPD=init.readCSVasPandas(biodivfiledir)
    
    siteIDList=list(biodivPD["SiteID"])
    num_sites=len(siteIDList)
    lon_ori_header="lon_original"                                              #header for original longitude
    lat_ori_header="lat_original"                                              #header for original latitude
    lon_calib_header="lon_HydroSHEDS"                                          #header for calibrated longitude
    lat_calib_header="lat_HydroSHEDS"                                          #header for calibrated latitude
    
    decay_func="exp"                                                           #exp, sph, matern15, matern25, gaussian
    env_var_name="discharge"                                                   #environmental variable to fit the baseline estimation
    bool_env_log=True                                                          #whether to log-transform the variable                                                         

#%%
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
    num_LULC_types=len(LULC_types)

#%%
    #set up distance list for estimation
    dist_list1=np.linspace(0.2,0.6,3).astype(np.float32)
    dist_list2=np.linspace(0.8,4,9).astype(np.float32)
    dist_list3=np.linspace(5,50,46).astype(np.float32)
    dist_list4=np.linspace(52,100,25).astype(np.float32)
    dist_list5=np.linspace(200,600,5).astype(np.float32)
    dist_list=np.concatenate((dist_list1,dist_list2,dist_list3,dist_list4,dist_list5),axis=0)
    num_dist_sim=len(dist_list)

#%%
    #read biodiversity value and environmental variable
    biodiv=np.array(biodivPD["biodiv"]).astype(np.float32).reshape([num_sites,1])
    env_var=np.array(biodivPD[env_var_name],dtype=np.float32).reshape([num_sites,1])
    if bool_env_log:
        env_var[env_var<0]=0
        env_var=np.log(env_var+1e-6)
        
    #get catchment file list for all sites    
    catch_file_list=[]
    for i_site in range(num_sites):
        catchdistfilename="site_ID_"+str(siteIDList[i_site]).zfill(2)+"_catchdist.tif"
        catchdistfiledir=catchfolderdir+os.sep+catchdistfilename
        if os.path.exists(catchdistfiledir):
            catch_file_list.append(catchdistfiledir)
            
#%%   
    #apply catchment location shifting for sites with large distance changes.
    bool_sitelocshift=True
    siteloc_thres=1
   
    C_mat=model.calc_C_mat(ras_LULC,LULC_CVT,biodivPD,siteIDList,catch_file_list,geoTrans_LULC,dist_list,LULC_types,env_var,\
                   num_sites,num_dist_sim,num_LULC_types,decay_func,siteloc_thres,bool_sitelocshift,\
                   lon_ori_header,lat_ori_header,lon_calib_header,lat_calib_header)
    
    LULC_values=model.calc_opt_params(C_mat,biodiv,dist_list,num_dist_sim,num_LULC_types)
        
#%%
        # write the results
    if not os.path.exists(resultfolderdir):
        os.makedirs(resultfolderdir)
    csvfilename="LULC_effect_"+decay_func+"_var_"+env_var_name+".csv"
    csvfiledir=resultfolderdir+os.sep+csvfilename
    init.writeArrayToCSV(LULC_values,["intcp"]+["V_lc"+str(LULC_type) for LULC_type in LULC_types]+["V_env","dist","adj_r2","neg2LogLik"],csvfiledir)
    print("ALL DONE.")    
