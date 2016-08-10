"""Script to test calling illastik."""

import ilastik_module as ilm 

ilm.runIlastik("../data/tifs/","../data/160804_toll10B_dapi.h5",classFile="classifiers/Dorsal_Dapi_alex2.ilp")