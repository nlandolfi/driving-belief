#!/bin/bash

set -e

./sim merge50_12_2_2passive
touch finished_merge50_12_2_2passive
./sim merge50_12_2_2active
touch finished_merge50_12_2_2active
./sim merge50_12_2_2passive_collab
touch finished_merge50_12_2_2passive_collab
./sim merge50_12_2_2active_collab
touch finished_merge50_12_2_2active_collab
./sim merge50_12_2_2passive_subtle
touch finished_merge50_12_2_2passive_subtle
./sim merge50_12_2_2active_subtle
touch finished_merge50_12_2_2active_subtle
./sim merge50_12_2_2passive_subtle_collab
touch finished_merge50_12_2_2passive_subtle_collab
./sim merge50_12_2_2active_subtle_collab
