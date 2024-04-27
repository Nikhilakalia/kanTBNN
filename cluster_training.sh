#!/bin/bash

source env_setup.sh

python3 -u training_run_config.py CFG_phll_only_cluster0 
python3 -u training_run_config.py CFG_phll_only_cluster1 
python3 -u training_run_config.py CFG_phll_only_cluster2 
python3 -u training_run_config.py CFG_phll_only_cluster3

python3 -u training_run_config.py CFG_duct_only_cluster0 
python3 -u training_run_config.py CFG_duct_only_cluster1 
python3 -u training_run_config.py CFG_duct_only_cluster2 
python3 -u training_run_config.py CFG_duct_only_cluster3
#python3 -u predict_cluster.py CFG_phll_only_clusterTBNN 

