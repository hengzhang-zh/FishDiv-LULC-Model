# **Project: Fish Diversity and Land Use Land Cover Modeling**

Authors: Heng Zhang (Hank) & [Prof. Florian Altermatt Lab](https://www.altermattlab.ch/)

Affiliations: Swiss Federal Institute of Aquatic Science and Technology (EAWAG/ETH) & University of Zurich (UZH)

Email: heng.zhang@eawag.ch

Date: 11/01/2023


# **Abstract**

Freshwater biodiversity is critically affected by human modifications of terrestrial land use and land cover (LULC). Yet, knowledge of the spatial extent and magnitude of LULC-aquatic biodiversity linkages is still surprisingly limited, impeding the implementation of optimal management strategies. Here, we compiled fish diversity data across a 160,000-km2 subtropical river catchment in Thailand characterized by exceptional biodiversity yet intense anthropogenic alterations, and attributed fish species richness and community composition to contemporary terrestrial LULC across the catchment. We estimated a spatial range of LULC effects extending up to about 20 km upstream from sampling sites, and explained nearly 60 % of the variance in the observed species richness, associated with major LULC categories including croplands, forest, and urban areas. We find that integrating both spatial range and magnitudes of LULC effects is needed to accurately predict fish species richness. Further, projected LULC changes showcase future gains and losses of fish species richness across the river network and offer a scalable basis for riverine biodiversity conservation and land management, allowing for potential mitigation of biodiversity loss in highly diverse yet data-deficient tropical to sub-tropical riverine habitats.


# **Platform and Packages**

This FishDiv-LULC modeling project was programmed with Python and CUDA for GPU computing. If you would like to run the program on CPU (can be time-consuming) or AMD GPU, you may translate the code to C++/OpenCL (please also inform us). Please make sure that you have already installed NVIDIA CUDA computing environment. We recommend using Anaconda python distribution. Please also make sure that the following modules have been successfully installed: 

_numpy, pandas, gdal, pycuda, pillow (PIL), scikit-learn, scipy_

For species-level modeling, we would recommend using the synthetic minority oversampling technique (SMOTE) to obtain class-balanced samples to fit the model. So, please install _imbalanced-learn_ module before running the species-level model. 


# **Code and Data**

Python and CUDA codes for catchment computing and FishDiv-LULC model are in the "code" folder. These codes can be executed in the following sequence. 

01_calc_catchment_HS.py: This code calculates the catchment for sampling sites, using HydroSHEDS flow direction map in this case. NVIDIA GPU is needed. 

02_estimate_LULC_effect_MLE.py: This code estimates the spatial range and magnitude of terrestrial LULC effects on fish species richness using maximum likelihood estimation (MLE). Only CPU is needed. 

03_estimate_LULC_effect_species_level_MLE.py: This code estimates the spatial range and magnitude of terrestrial LULC effects on fish species/habitat distribution based on species-level modeling. The associated LULC type for fish species was also determined with this species-level modeling approach. Only CPU is needed. 

04_calc_LULC_effect_map.py: This code calculates and produces a map of terrestrial LULC effect, based on flow direction, flow accumulation maps and the estimated optimal parameters from the above step. NVIDIA GPU is needed. 

05_pred_river_biodiv_map.py: This code projects riverine fish species richness in major river channels, using flow direction, flow accumulation maps and the estimated optimal parameters. NVIDIA GPU is needed. 

06_pred_river_species_p_a.py: This code predicts distribution/habitat of fish species in major river channels. The output is a csv file showing presence/absence of each fish species with indices of major river channel pixels in the first two columns (to save space on the disc). You may need to covert the csv file to distribution maps in further steps. NVIDIA GPU is needed. 

All python functions and CUDA kernel functions are in the "code/src" folder. 


eDNA-derived fish diversity (including presence/absence for four fish species as a demo) together with sampling site coordinates and river discharge are in the "data/eDNA" folder. You may find HydroSHEDS flow direction map, flow accumulation map, elevation map, and interpolated river discharge for major river channels in the "data/RS/catch_data" folder. Lastly, the land cover map we used in this study (European Space Agency Climate Change Initiative (ESA CCI) land cover) together with recoding table are in the "data/RS/land_cover" folder. 


# **Reference**

H. Zhang, R. Blackman, R. Furrer, M. Osathanunkul, J. Brantschen, C. Di Muri, L. R. Harper, B. Hänfling, P. Niklaus, L. Pellissier, M. Schaepman, S. Zong, F. Altermatt*, 2023, Terrestrial land cover shapes fish diversity in major subtropical rivers. 
BioRxiv Link: 
