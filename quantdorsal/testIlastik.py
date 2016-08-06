"""Script to test calling illastik."""

import ilastik_module as ilm 

ilm.runIlastik("../data/160804_toll10B_dapi.zip.lif","../data/160804_toll10B_dapi.h5",classFile="classifiers/Dorsal_Dapi_1.ilp")