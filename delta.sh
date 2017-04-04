#!/bin/bash

set -e

./sim subtle_merge_passive
./sim subtle_merge_active
./sim subtle_merge_passive_collab
./sim subtle_merge_active_collab

./sim subtle_hold_passive
./sim subtle_hold_active
./sim subtle_hold_passive_collab
./sim subtle_hold_active_collab
