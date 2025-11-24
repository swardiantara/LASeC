#!/bin/bash
datasets=( Apache Android )
# datasets=( MultiUnique )
models=( agglomerative birch )
# embeddings=( MultiSource-full-crk0-m0.5-e5-b128-L6 MultiSource-full-crk1-m0.5-e5-b128-L6 MultiSource-full-cdk1-m0.5-e5-b128-L6 MultiSource-full-crk3-m0.5-e5-b128-L6 MultiSource-full-cdk3-m0.5-e5-b128-L6 MultiSource-full-crk5-m0.5-e5-b128-L6 MultiSource-full-cdk5-m0.5-e5-b128-L6 MultiSource-full-crk10-m0.5-e5-b128-L6 MultiSource-full-cdk10-m0.5-e5-b128-L6 ) # one-crk10-m0.5-e5-b128-L6 one-crk10-m0.05-e2-b128-L6 one-cdk10-m0.5-e5-b16-v1 
thresholds=( 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.9 0.95 1 )

# embeddings=( MultiSource-full-crk1-m0.5-e5-b128-L6 MultiSource-full-cdk1-m0.5-e5-b128-L6 MultiSource-full-crk3-m0.5-e5-b128-L6 MultiSource-full-cdk3-m0.5-e5-b128-L6 MultiSource-full-crk5-m0.5-e5-b128-L6 MultiSource-full-cdk5-m0.5-e5-b128-L6 MultiSource-full-crk10-m0.5-e5-b128-L6 MultiSource-full-cdk10-m0.5-e5-b128-L6 ) # one-crk10-m0.5-e5-b128-L6 one-crk10-m0.05-e2-b128-L6 one-cdk10-m0.5-e5-b16-v1 
# for embedding in "${embeddings[@]}"; do
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for threshold in "${thresholds[@]}"; do
            python -m src.lasec --dataset "$dataset" --model "$model" --threshold "$threshold" --output_dir grid-loso --held_out
        done
    done
done
# done