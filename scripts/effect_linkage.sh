#!/bin/bash

datasets=( Apache Android BGL Hadoop HDFS HealthApp HPC Linux Mac OpenSSH OpenStack Proxifier Spark Thunderbird Windows Zookeeper MultiSource MultiUnique )
linkages=( ward complete single )
embeddings=( MultiSource-full-crk2-m0.5-e5-b128-L6 ) # one-crk10-m0.5-e5-b128-L6 one-crk10-m0.05-e2-b128-L6 one-cdk10-m0.5-e5-b16-v1 
thresholds=( 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 )
for embedding in "${embeddings[@]}"; do
    for dataset in "${datasets[@]}"; do
        for linkage in "${linkages[@]}"; do
            for threshold in "${thresholds[@]}"; do
                python -m src.lasec --dataset "$dataset" --model agglomerative --linkage "$linkage" --embedding "$embedding" --threshold "$threshold" --output_dir effect-linkage
                # python -m src.lasec --dataset "$dataset" --model "$model" --embedding "$embedding" --threshold "$threshold" --output_dir effect-k --held_out
            done
        done
    done
done