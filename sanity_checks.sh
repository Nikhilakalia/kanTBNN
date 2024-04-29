#!/bin/bash

source env_setup.sh

#python3 -u label_injections.py | tee log.label_injections

python3 -u training_run_config.py CFG_sanity_check_fp_memorization 
#python3 -u training_run_config.py CFG_sanity_check_fp_only 
#python3 -u training_run_config.py CFG_sanity_check_fp_only_norealiz 

python3 -u training_run_config.py CFG_sanity_check_phll_memorization
#python3 -u training_run_config.py CFG_sanity_check_phll_only
#python3 -u training_run_config.py CFG_sanity_check_phll_only_norealiz

python3 -u training_run_config.py CFG_sanity_check_duct_memorization
#python3 -u training_run_config.py CFG_sanity_check_duct_only
#python3 -u training_run_config.py CFG_sanity_check_duct_only_norealiz
