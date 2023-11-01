# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:33:10 2023

@author: Heng Zhang
"""

import copy
import numpy as np

from PIL import Image, ImageDraw

import src.processgeotiff4 as ptf

#generate destined points (water may flow through) based on given sampling points
def genDstPoints(point_coord,GeoTransform,radius_dist=100):
    [pX, pY] = ptf.world2Pixel(GeoTransform, point_coord[0], point_coord[1])  
    p_size=max(abs(GeoTransform[1]),abs(GeoTransform[5]))
    radius_num=int(radius_dist/p_size)
    dst_points=[]
    for j in range(pY-radius_num,pY+radius_num+1):
        for i in range(pX-radius_num,pX+radius_num+1):
            dist=np.sqrt(np.square(j-pY)+np.square(i-pX))
            if dist<=radius_num:
                dst_points.append([j,i])
    return dst_points

#initialize catchment mask and catchment distance data
def initCatchDist(dst_points,nrow,ncol,default_dist_value):
    catchdist=np.zeros([nrow,ncol],dtype=np.float32)
    for dp in dst_points:
        catchdist[dp[0],dp[1]]=default_dist_value   #here -1 stands for the outlet of catchment
    return catchdist

def calcLonLatDist(p1_x,p1_y,p2_x,p2_y,GeoTransform):
 	#p1_lonlat_x = (GeoTransform[0] + p1_x*GeoTransform[1])*np.pi / 180
 	p1_lonlat_y = (GeoTransform[3] + p1_y*GeoTransform[5])*np.pi / 180
 	#p2_lonlat_x = (GeoTransform[0] + p2_x*GeoTransform[1])*np.pi / 180
 	p2_lonlat_y = (GeoTransform[3] + p2_y*GeoTransform[5])*np.pi / 180
 	
 	dlat = (p2_y-p1_y)*GeoTransform[1]*np.pi/180
 	dlon = (p2_x-p1_x)*GeoTransform[5]*np.pi/180
 	a = np.sin(dlat / 2)*np.sin(dlat / 2) + np.cos(p1_lonlat_y)*np.cos(p2_lonlat_y) * np.sin(dlon / 2)*np.sin(dlon / 2)
 	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
 	return 6371*c

def calcPixelAreaMat(nrow,ncol,GeoTransform):
    pixel_area_rows=np.zeros(nrow,dtype=np.float32)
    for j in range(nrow):
        [p1_x,p1_y,p2_x,p2_y]=[0,j,0,j+1]        
        dy=calcLonLatDist(p1_x,p1_y,p2_x,p2_y,GeoTransform)
        [p1_x,p1_y,p2_x,p2_y]=[0,j,1,j]        
        dx=calcLonLatDist(p1_x,p1_y,p2_x,p2_y,GeoTransform)
        pixel_area_rows[j]=dy*dx
    A=np.zeros([nrow,ncol],dtype=np.float32)
    for i in range(ncol):
        A[:,i]=pixel_area_rows[:]
    return A

def calcPixelDXDYArray(nrow,ncol,GeoTransform):
    DX=np.zeros(nrow,dtype=np.float32)
    DY=np.zeros(ncol,dtype=np.float32)    
    for j in range(nrow):     
        [p1_x,p1_y,p2_x,p2_y]=[0,j,1,j]        
        DX[j]=calcLonLatDist(p1_x,p1_y,p2_x,p2_y,GeoTransform)  
    for i in range(ncol):
        [p1_x,p1_y,p2_x,p2_y]=[i,0,i,1]        
        DY[i]=calcLonLatDist(p1_x,p1_y,p2_x,p2_y,GeoTransform)
    return [DX,DY]

def clipRasterBBOX(ras1,ras2,geoTrans):
    cd_row_sum=np.sum(ras2,axis=1)
    cd_col_sum=np.sum(ras2,axis=0)
    row_start=np.where(cd_row_sum>0)[0][0]
    row_end=np.where(cd_row_sum>0)[0][-1]
    col_start=np.where(cd_col_sum>0)[0][0]
    col_end=np.where(cd_col_sum>0)[0][-1]
    
    ras1=np.ascontiguousarray(ras1[row_start:row_end,col_start:col_end])
    ras2=np.ascontiguousarray(ras2[row_start:row_end,col_start:col_end])
    
    geoTrans[0]=geoTrans[0]+geoTrans[1]*col_start
    geoTrans[3]=geoTrans[3]+geoTrans[5]*row_start
    
    nrow=int(row_end-row_start)
    ncol=int(col_end-col_start)  
    return [ras1,ras2,geoTrans,nrow,ncol]

def calcShiftVector(ref_x,ref_y,tar_x,tar_y,pixel_reso):
    shift_x=tar_x-ref_x
    shift_y=tar_y-ref_y
    coord_shift=np.sqrt(shift_x*shift_x+shift_y*shift_y)
    num_pixel_shift=coord_shift/pixel_reso
    siteShifts=np.array([shift_x,shift_y,num_pixel_shift]).transpose()
    return siteShifts

def shiftRasterGeoTrans(shift_x,shift_y,geoTrans):
    geoTrans_shift=copy.deepcopy(geoTrans)
    pixel_shift_x=round(shift_x/geoTrans[1])
    pixel_shift_y=round(shift_y/geoTrans[5])
    
    geoTrans_shift[0]=geoTrans[0]+geoTrans[1]*pixel_shift_x
    geoTrans_shift[3]=geoTrans[3]+geoTrans[5]*pixel_shift_y
    return geoTrans_shift

def clipRaster(ras,geoTrans_ras,geoTrans_clip,nrow,ncol):
    #clip FD to match buffer of sampling sites
    [i_col_start, i_row_start]=ptf.world2Pixel(geoTrans_ras, geoTrans_clip[0], geoTrans_clip[3])
    [i_col_end, i_row_end]=ptf.world2Pixel(geoTrans_ras, geoTrans_clip[0]+ncol*geoTrans_clip[1], geoTrans_clip[3]+nrow*geoTrans_clip[5])
    
    geoTrans_out=copy.deepcopy(geoTrans_ras)
    geoTrans_out[0]=geoTrans_ras[0]+geoTrans_ras[1]*i_col_start
    geoTrans_out[3]=geoTrans_ras[3]+geoTrans_ras[5]*i_row_start
    
    ras_out=np.ascontiguousarray(ras[i_row_start:i_row_end,i_col_start:i_col_end])

    return [ras_out,geoTrans_out]

def resizeImage(srcImage,dst_shape):
    img=Image.fromarray(srcImage)
    # dstImage=np.array(img.resize(dst_shape,Image.Resampling.NEAREST),dtype=np.float32)
    dstImage=np.array(img.resize(dst_shape,Image.NEAREST),dtype=np.float32)
    return dstImage

def cvtLCTypes(ras_LC,LC_CVT,LC_header_in,LC_header_out,rm_list_out=[]):
    LC_types_in=list(LC_CVT[LC_header_in])
    LC_types_out=list(LC_CVT[LC_header_out])
    ras_LC_cvt=np.zeros_like(ras_LC,dtype=np.int16)+999  #zero is default (No Data)
    for i_type in range(len(LC_types_in)):
        lc_type_in=LC_types_in[i_type]
        lc_type_out=LC_types_out[i_type]
        if lc_type_out in rm_list_out:
            continue
        ras_LC_cvt[ras_LC==lc_type_in]=lc_type_out
    return ras_LC_cvt

def extRiverPixels(FA,FA_thres):
    bool_RivPix=np.zeros_like(FA,dtype=np.bool8)
    bool_RivPix[FA>=FA_thres]=True
    river_pixels=np.matrix(np.int32(np.argwhere(bool_RivPix==True)))
    return river_pixels