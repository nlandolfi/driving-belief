#!/bin/bash

set -e

./sim mergex_passive
touch xpassive
./sim mergex_active
touch xactive
./sim mergex_passive_collab
touch xpassive_collab
./sim mergex_active_collab
touch xactive_collab

./sim speed_12_2_2passive
touch spassive
./sim speed_12_2_2active
touch sactive
./sim speed_12_2_2passive_collab
touch spassive_c
./sim speed_12_2_2active_collab
touch sactive_c
