#!/bin/bash

source ./tf23/bin/activate
export LD_LIBRARY_PATH=/srv/storage/talc@talc-data.nancy/multispeech/calcul/users/dfohr/cuda/lib64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/srv/storage/talc@talc-data.nancy/multispeech/calcul/users/dfohr/cuda/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/srv/storage/talc@talc-data.nancy/multispeech/calcul/users/dfohr/cuda/include:$CPLUS_INCLUDE_PATH


python dotrain_semantique_kerasBert_acclmscore_original.py bert_lstm_sansunk_aclms_cmpwithbesthypo_100loc    > out_100locs_sansunk_aclms_cmpwithbesthypo  2>&1

deactivate