# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:56:50 2019

@author: ZH
"""

import copy
import numpy as np
from osgeo import gdal,osr
from PIL import Image, ImageDraw

#convert coordinates (x,y) to indices (idx,idy) of data matrix
def world2Pixel(geoMatrix,x,y):
    ulX=geoMatrix[0]                 #lon (X) coordinate of the upper-left corner 
    ulY=geoMatrix[3]                 #lat (Y) coordinate of the upper-left corner
    xDist=geoMatrix[1]               #distance of each pixel at longitudinal direction
    yDist=geoMatrix[5]               #distance of each pixel at latitudinal direction
#    rtnX=geoMatrix[2]
#    rtnY=geoMatrix[4]
    idx=int((x-ulX)/xDist)
    idy=int((ulY-y)/abs(yDist))  
    return (idx, idy)    

def readTiffAsNumpy(TiffList,datatype='Float32'):
#    print("Reading GeoTiff files...")
    total=len(TiffList)
    tmpfiledir=TiffList[0]
    tmp=gdal.Open(tmpfiledir)
    ncol=tmp.RasterXSize
    nrow=tmp.RasterYSize
    Driver=tmp.GetDriver()
    GeoTransform=np.array(tmp.GetGeoTransform())
    Proj=tmp.GetProjection()
    if datatype=="Int8":
        dtp=np.int8
    elif datatype=="UInt8":
        dtp=np.uint8
    elif datatype=="Int16":
        dtp=np.int16
    elif datatype=="UInt16":
        dtp=np.uint16
    elif datatype=="Int32":
        dtp=np.int32
    elif datatype=="UInt32":
        dtp=np.uint32
    elif datatype=="Float32":
        dtp=np.float32
    elif datatype=="Float64":
        dtp=np.float64
    else:
        print("Data type not listed! Please choose from the bellowing:")
        print("Int8 UInt8 Int16 UInt16 UInt16 Int32 UInt32 Float32 Float64")    
    OriData=np.zeros([nrow,ncol,total],dtype=dtp)
    for i in range(total):
#        print("reading: %s"%TiffList[i])
        data=gdal.Open(TiffList[i])
        raster=data.ReadAsArray().astype(dtp)
        try:
            OriData[:,:,i]=raster
        except:
            OriData[:,:,i]=np.zeros([nrow,ncol],dtype=dtp)
    return [OriData,Driver,GeoTransform,Proj,nrow,ncol]

def getBlockRasterExtent(filelist):
    ras_bbox=np.array([9999999999,-9999999999,9999999999,-9999999999],dtype=np.float32)
    for i in range(len(filelist)):
        blockmapfiledir=filelist[i]
        tmp=gdal.Open(blockmapfiledir)
        geoTrans_tmp=tmp.GetGeoTransform()
        minX = geoTrans_tmp[0]
        maxY = geoTrans_tmp[3]
        maxX = minX + geoTrans_tmp[1] * tmp.RasterXSize
        minY = maxY + geoTrans_tmp[5] * tmp.RasterYSize
        ras_bbox[0]=min(ras_bbox[0],minX)
        ras_bbox[1]=max(ras_bbox[1],maxX)
        ras_bbox[2]=min(ras_bbox[2],minY)
        ras_bbox[3]=max(ras_bbox[3],maxY)
        del tmp
    geoTrans=np.array(geoTrans_tmp)
    geoTrans[0]=ras_bbox[0]
    geoTrans[3]=ras_bbox[3]
    ncol=int((ras_bbox[1]-ras_bbox[0])/abs(geoTrans[1]))
    nrow=int((ras_bbox[3]-ras_bbox[2])/abs(geoTrans[5]))
    return [geoTrans,ras_bbox,nrow,ncol]

def clipRas2Rect(ras,geoTrans):
    cd_row_sum=np.sum(ras,axis=1)
    cd_col_sum=np.sum(ras,axis=0)
    row_start=np.where(cd_row_sum>0)[0][0]
    row_end=np.where(cd_row_sum>0)[0][-1]
    col_start=np.where(cd_col_sum>0)[0][0]
    col_end=np.where(cd_col_sum>0)[0][-1]
    
    ras=np.ascontiguousarray(ras[row_start:row_end,col_start:col_end])
    
    geoTrans=list(geoTrans)
    geoTrans[0]=geoTrans[0]+geoTrans[1]*col_start
    geoTrans[3]=geoTrans[3]+geoTrans[5]*row_start
    
    nrow=int(row_end-row_start)
    ncol=int(col_end-col_start)  
    return [ras,geoTrans,nrow,ncol]

def calcRowColIndexBBOX(ras_shape,geoTrans,ras_bbox):
    row_start=max(0,int(0.5+(ras_bbox[3]-geoTrans[3])/geoTrans[5]))
    row_end=min(ras_shape[0],int(0.5+(ras_bbox[2]-geoTrans[3])/geoTrans[5]))
    col_start=max(0,int(0.5+(ras_bbox[0]-geoTrans[0])/geoTrans[1]))
    col_end=min(ras_shape[1],int(0.5+(ras_bbox[1]-geoTrans[0])/geoTrans[1]))
    if row_end==row_start:
        row_end+=1
    if col_end==col_start:
        col_end+=1    
    return [row_start,row_end,col_start,col_end]    

def clipRasBBOX(ras,geoTrans,ras_bbox):
    ras_shape=ras.shape
    [row_start,row_end,col_start,col_end]=calcRowColIndexBBOX(ras_shape,geoTrans,ras_bbox)    
    ras_clip=np.ascontiguousarray(ras[row_start:row_end,col_start:col_end])
    
    geoTrans_clip=list(copy.deepcopy(geoTrans))
    geoTrans_clip[0]=geoTrans[0]+geoTrans[1]*col_start
    geoTrans_clip[3]=geoTrans[3]+geoTrans[5]*row_start
    
    nrow=int(row_end-row_start)
    ncol=int(col_end-col_start)  
    return [ras_clip,geoTrans_clip,nrow,ncol]

def resizeImage(srcImage,dst_shape):
    img=Image.fromarray(srcImage)
    dstImage=np.array(img.resize(dst_shape,Image.Resampling.NEAREST),dtype=np.float32)
    return dstImage

def createTIFFDriverFromEPSG(EPSG_ID):
    driver=gdal.GetDriverByName("GTiff")
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(EPSG_ID)   #CH1903/LV03
    proj = proj.ExportToWkt()
    return [driver,proj]

def writeNumpyToTiff(TargetData,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,filedirto,datatype='Float32'):
    if datatype=='Int8':
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Byte)
        TargetData=TargetData.astype(np.int8)
    elif datatype=='Int16':        
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Int16)
        TargetData=TargetData.astype(np.int16)
    elif datatype=='Int32':
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Int32)
        TargetData=TargetData.astype(np.int32)  
    elif datatype=='UInt16':        
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_UInt16)
        TargetData=TargetData.astype(np.uint16)
    elif datatype=='UInt32':
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_UInt32)
        TargetData=TargetData.astype(np.uint32)  
    elif datatype=='Float32':        
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Float32)
        TargetData=TargetData.astype(np.float32)
    elif datatype=='Float64':
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Float64)
        TargetData=TargetData.astype(np.float64)        
    else:
        print("Data type not listed! Please choose from the bellowing:")
        print("Int8  Int16  Int32  UInt16  UInt32  Float32  Float64")
    output.SetGeoTransform(GeoTransform)
    output.SetProjection(Proj)
    outBand=output.GetRasterBand(1)
#    outBand.SetNoDataValue(nanDefault)    
    outBand.WriteArray(TargetData,0,0)
    outBand.FlushCache()
    
