# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:28:32 2023

@author: Heng Zhang
"""

import numpy as np
import pandas as pd

def readCSVasPandas(filedir):
    try: 
        pdData=pd.DataFrame(pd.read_csv(filedir,header=0))
    except: 
        pdData=pd.DataFrame(pd.read_csv(filedir,header=0,encoding= 'unicode_escape'))
    return pdData
    
def getListFrompdDataSet(pdData,header):
    pdList=list(pdData[header])
    return pdList

def writeArrayToCSV(Array,ArrayNames,filedirto):
    if not Array.shape[1]==len(ArrayNames):
        print("Array Dim != len(ArrayNames)")
        return
    save=pd.DataFrame({"ID":np.linspace(1,Array.shape[0],Array.shape[0]).astype(np.int32)})
    for i in range(Array.shape[1]):
        pdtmp=pd.DataFrame({ArrayNames[i]:Array[:,i]})
        save=pd.concat([save,pdtmp],axis=1)
    save.to_csv(filedirto,index=False,header=True)