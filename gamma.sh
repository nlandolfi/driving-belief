#!/bin/bash

set -e

./sim exit_passive
./sim exit_active
./sim exit_passive_collab
./sim exit_active_collab

./sim merge_passive
./sim merge_active
./sim merge_passive_collab
./sim merge_active_collab
