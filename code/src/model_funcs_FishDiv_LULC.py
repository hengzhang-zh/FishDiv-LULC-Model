# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:35:14 2023

@author: Heng Zhang
"""

import os
import copy
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.linear_model import TweedieRegressor
#the following three modules are only used in species-level modeling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from imblearn.over_sampling import SMOTE 

import src.catch_funcs as catch
import src.processgeotiff4 as ptf


def S_LC_exp(FD_vector,alpha,beta,pixel_areas):
    s=pixel_areas*beta*np.exp(-1*alpha*FD_vector)
    return np.sum(s)

def S_LC_sph(FD_vector,theta1,theta2,pixel_areas):
    bool_range=FD_vector<=theta1
    s=np.zeros_like(FD_vector)
    s[bool_range]=pixel_areas[bool_range]*theta2*(1-1.5*(FD_vector[bool_range]/theta1)+0.5*(FD_vector[bool_range]/theta1)**3)
    return np.sum(s)

def S_LC_Gaussian(FD_vector,alpha,beta,pixel_areas):
    s=pixel_areas*beta*np.exp(-1*alpha*alpha*FD_vector*FD_vector)
    return np.sum(s)

def S_LC_Matern15(FD_vector,alpha,beta,pixel_areas):
    s=pixel_areas*beta*(1+np.sqrt(3)*alpha*FD_vector)*np.exp(-1*np.sqrt(3)*alpha*FD_vector)
    return np.sum(s)

def S_LC_Matern25(FD_vector,alpha,beta,pixel_areas):
    s=pixel_areas*beta*(1+np.sqrt(5)*alpha*FD_vector+5/3*alpha*alpha*FD_vector*FD_vector)*np.exp(-1*np.sqrt(5)*alpha*FD_vector)
    return np.sum(s)

def calcOptTSSDiv(p_a,p_a_prob,intv=0.01):
    divs=np.linspace(intv,1,int(1/intv),dtype=np.float32)
    TSSs=np.zeros_like(divs,dtype=np.float32)
    p_prob=p_a_prob[:,1]
    for i in range(int(1/intv)):
        div=divs[i]
        bool_p=1*(p_prob>div)
        a=np.sum((p_a==1)&(bool_p==1))
        b=np.sum((p_a==0)&(bool_p==1))
        c=np.sum((p_a==1)&(bool_p==0))
        d=np.sum((p_a==0)&(bool_p==0))
        sensi=a/(a+c)
        speci=d/(b+d)
        TSSs[i]=sensi+speci-1
    idx_max=np.argmax(TSSs)
    return [divs[idx_max],TSSs[idx_max]]

def calc_neg2LogLik(params,X,y,num_sites,num_LULC_types):
    a=params[0]
    b=params[1+num_LULC_types]
    sigma2=params[2+num_LULC_types]
    V=params[1:(1+num_LULC_types)]

    mu=a*X[:,0]+b*X[:,1+num_LULC_types]+np.dot(X[:,1:(1+num_LULC_types)],V)
    resid=y-mu
    LL=-0.5*num_sites*(np.log(sigma2)+np.log(2*np.pi))-0.5*np.sum(resid*resid/sigma2)
    return -2*LL

def calc_C_site(i_site,C,ras_LULC_cvt,FD,A,env_var,dist_list,LULC_types,num_dist_sim,num_LULC_types,decay_func):
    # C=np.zeros([num_sites,num_LC_types+2])
    C[:,i_site,0]=1
    
    FD_mask=np.zeros_like(FD,dtype=np.bool_)    
    FD_mask[FD>0]=True
    
    for i_dist in range(num_dist_sim):
        dist=dist_list[i_dist]
        for i_type in range(num_LULC_types):
            LULC_type=LULC_types[i_type]
            bool_LULC_type=np.zeros_like(ras_LULC_cvt,dtype=np.bool_)
            bool_LULC_type[ras_LULC_cvt==LULC_type]=True
            bool_LULC_type=bool_LULC_type&FD_mask
            FD_vector=FD[bool_LULC_type]
            pixel_areas=A[bool_LULC_type]
            
            if decay_func=="exp":
                alpha=3/dist
                beta=alpha
                S_lc=S_LC_exp(FD_vector,alpha,beta,pixel_areas)
            elif decay_func=="sph":
                theta1=dist
                theta2=(8/3/theta1)
                S_lc=S_LC_sph(FD_vector,theta1,theta2,pixel_areas)
            elif decay_func=="gaussian":
                alpha=3/dist
                beta=2*np.sqrt(alpha*alpha/np.pi/3)
                S_lc=S_LC_Gaussian(FD_vector,alpha/np.sqrt(3),beta,pixel_areas)
            elif decay_func=="matern15":
                alpha=3/dist
                beta=(np.sqrt(3)*alpha)/2
                S_lc=S_LC_Matern15(FD_vector,alpha,beta,pixel_areas)
            elif decay_func=="matern25":
                alpha=3/dist
                beta=(3*np.sqrt(5)*alpha)/8
                S_lc=S_LC_Matern15(FD_vector,alpha,beta,pixel_areas)
            else:
                print("decay function method error!")
            C[i_dist,i_site,i_type+1]=S_lc
            C[i_dist,i_site,num_LULC_types+1]=env_var[i_site,0]
        
def calc_C_mat(ras_LULC,LULC_CVT,biodivPD,siteIDList,catch_file_list,geoTrans_LULC,dist_list,LULC_types,env_var,\
               num_sites,num_dist_sim,num_LULC_types,decay_func,siteloc_thres,bool_sitelocshift,\
               lon_ori_header,lat_ori_header,lon_calib_header,lat_calib_header):
    #shift flow direction map location
    if bool_sitelocshift:
        siteLocShifts=catch.calcShiftVector(biodivPD[lon_ori_header],biodivPD[lat_ori_header],\
                                            biodivPD[lon_calib_header],biodivPD[lat_calib_header],\
                                            geoTrans_LULC[1])
    
    #calculate C matrix for optimization
    C_mat=np.zeros([num_dist_sim,num_sites,num_LULC_types+2],dtype=np.float32)
    for i_site in range(num_sites):
        siteID=siteIDList[i_site]
        print("calculating C matrix site ID: %s"%(siteID))        
        catchdistfiledir=catch_file_list[i_site]
        [catch_FD,driver_FD,geoTrans_FD,proj_FD,nrow_FD,ncol_FD]=ptf.readTiffAsNumpy([catchdistfiledir],datatype='Float32')
        catch_FD=catch_FD[:,:,0]
        #shift location of flow direction map
        if bool_sitelocshift:
            [shift_x,shift_y,num_pixel_shift]=siteLocShifts[i_site,:]
            if num_pixel_shift>siteloc_thres:
                geoTrans_FD=catch.shiftRasterGeoTrans(shift_x, shift_y, geoTrans_FD)
        #clip LULC map
        [ras_LULC_catch,geoTrans_LULC_catch]=catch.clipRaster(ras_LULC,geoTrans_LULC,geoTrans_FD,nrow_FD,ncol_FD)         
        ras_catch_dim=ras_LULC_catch.shape
        if not np.sum(ras_LULC_catch):
            C_mat[:,i_site,0]=-1
            continue
        A_catch=catch.calcPixelAreaMat(ras_catch_dim[0],ras_catch_dim[1],geoTrans_LULC_catch)
        catch_FD=catch.resizeImage(catch_FD,[ras_catch_dim[1],ras_catch_dim[0]])
        ras_LULC_catch_cvt=catch.cvtLCTypes(ras_LULC_catch,LULC_CVT,"ESACCI_VALUE","CUSTOM_VALUE",[])        
        #calculate C values for one site
        calc_C_site(i_site,C_mat,ras_LULC_catch_cvt,catch_FD,A_catch,env_var,dist_list,LULC_types,\
                    num_dist_sim,num_LULC_types,decay_func)
    return C_mat

def calc_opt_params(C_mat,biodiv,dist_list,num_dist_sim,num_LULC_types):
    LULC_values=np.zeros([num_dist_sim,num_LULC_types+5])    
    #remove all-zero records
    bool_site=C_mat[0,:,0]!=-1
    C_mat=np.ascontiguousarray(C_mat[:,bool_site,:])
    biodiv=biodiv[bool_site,:]
    num_sites=np.sum(bool_site)
    
    #calculate num_LULC_exist
    bool_LULC=np.sum(C_mat[num_dist_sim-1,:,1:(num_LULC_types+1)],axis=0)>0
    num_LULC_exist=np.sum(bool_LULC)
    
    biodiv_mean=np.mean(biodiv)
    TSS=np.sum((biodiv-biodiv_mean)**2)
    
    biodiv_B0=np.zeros([num_sites,1],dtype=np.float32)     

    #optimization
    for i_dist in range(num_dist_sim):#%%
        dist=dist_list[i_dist]              
        C=np.ascontiguousarray(C_mat[i_dist,:,:])
        
        #step 6: regress C and biodiv to estimate LC_Values (OLS)
        lm_model=np.linalg.lstsq(C, biodiv-biodiv_B0, rcond=None)
        lm_coefs=lm_model[0]
        biodiv_pred=np.dot(C,lm_coefs)+biodiv_B0
        resid=biodiv-biodiv_pred
        RSS=np.sum(resid*resid)  
        sigma2=np.sum(resid*resid)/num_sites
        coefs=lm_coefs[:,0]
        neg2LL=num_sites*(np.log(sigma2)+np.log(2*np.pi))+np.sum(resid*resid/sigma2)

        # #step 6: estimate parameters using maximum likelihood estimation (MLE)
        # params=np.concatenate((np.array([30,3,-2]),np.repeat(2, num_LULC_types)),axis=0)
        # # params=params+np.random.uniform(low=-0.1,high=0.1,size=num_LULC_types+3)
        # # params[0:(num_LULC_types+2)]=coefs 
        # # params[num_LULC_types+2]=sigma2
        # y=biodiv-biodiv_B0
        # y=np.float32(y.flatten())     
        # opt_result=opt.minimize(calc_neg2LogLik, params, args=(C,y,num_sites,num_LULC_types), method='L-BFGS-B')             
        # mle_coefs=opt_result.x[0:(2+num_LULC_types)]
        # mle_fvalue=opt_result.fun
        # neg2LL=mle_fvalue
        # biodiv_pred=np.dot(C,mle_coefs)+biodiv_B0.flatten()
        # RSS=np.sum((biodiv_pred-biodiv.flatten())**2)
        # coefs=mle_coefs
            
        adj_r2=1-(RSS/(num_sites-num_LULC_exist-2-1))/(TSS/(num_sites-1))
        LULC_values[i_dist,(num_LULC_types+2-len(coefs)):(num_LULC_types+2)]=coefs      
        LULC_values[i_dist,num_LULC_types+2]=dist
        LULC_values[i_dist,num_LULC_types+3]=adj_r2
        LULC_values[i_dist,num_LULC_types+4]=neg2LL
        print("simulation no. %d / %d (dist = %.1f km),  r-sq = %.4f, -2LL = %.2f\n"%(i_dist,num_dist_sim,dist,adj_r2,neg2LL))
    return LULC_values

def calc_opt_params_species_level(C_mat,biodiv,dist_list,num_dist_sim,num_LULC_types):
    LULC_values=np.zeros([num_dist_sim,num_LULC_types+6])    
    #remove all-zero records
    bool_site=C_mat[0,:,0]!=-1
    C_mat=np.ascontiguousarray(C_mat[:,bool_site,:])
    biodiv=biodiv[bool_site,:].ravel()
    num_sites=np.sum(bool_site)
    if np.sum(biodiv)==num_sites or np.sum(biodiv)==0:
        print("model for this species cannot be built up. \n")
        return    
    for i_dist in range(num_dist_sim):
        dist=dist_list[i_dist]    
        C=np.ascontiguousarray(C_mat[i_dist,:,:])
        #SMOTE to balance classes
        k_neighbors=min(np.min(np.bincount(biodiv))-1,5)
        if k_neighbors>0:
            sm=SMOTE(k_neighbors=k_neighbors)
            [C_res,biodiv_res]=sm.fit_resample(C, biodiv)        
        else:                
            idx=np.where(biodiv==1)[0][0]
            C_app=np.zeros([num_sites+1,num_LULC_types+2])
            biodiv_app=np.zeros(num_sites+1)
            C_app[0:num_sites,:]=C
            biodiv_app[0:num_sites]=biodiv
            C_app[num_sites,:]=C[idx,:]
            biodiv_app[num_sites]=biodiv[idx]
            k_neighbors=1
            sm=SMOTE(k_neighbors=k_neighbors)
            [C_res,biodiv_res]=sm.fit_resample(C_app, biodiv_app)
            
        clf = LogisticRegression(fit_intercept=True).fit(C_res, biodiv_res)
        coefs=clf.coef_.ravel()
        coef_intcp=clf.intercept_
        biodiv_pred_prob=clf.predict_proba(C)
        [p_a_thres,p_a_TSS]=calcOptTSSDiv(biodiv,biodiv_pred_prob,intv=0.01)
        
        biodiv_pred=1*(biodiv_pred_prob[:,1]>=p_a_thres)
        # ACC=np.sum(biodiv_pred==biodiv.ravel())/num_sites
        KAPPA=cohen_kappa_score(biodiv_pred,biodiv)
        
        LULC_values[i_dist,0]=coef_intcp
        LULC_values[i_dist,(num_LULC_types+2-len(coefs)):(num_LULC_types+2)]=coefs      
        LULC_values[i_dist,num_LULC_types+2]=dist
        LULC_values[i_dist,num_LULC_types+3]=p_a_thres
        LULC_values[i_dist,num_LULC_types+4]=p_a_TSS
        LULC_values[i_dist,num_LULC_types+5]=KAPPA
        print("simulation no. %d / %d (dist = %.1f), thres = %.2f, metric = %.3f\n"%(i_dist,num_dist_sim,dist,p_a_thres,p_a_TSS))
    return LULC_values