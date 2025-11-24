# !bash

# fine-tuning embedding with leave-one-source-out strategy for robustness test
datasets=( Apache Android BGL Hadoop HDFS HealthApp HPC Linux Mac OpenSSH OpenStack Proxifier Spark Thunderbird Windows Zookeeper )
model="all-MiniLM-L6-v2"
sampling="distance"
k=3

for dataset in "${datasets[@]}"; do
    python -m src.train_embedding --initial_model_path "$model" --sampling_strategy "$sampling" --k "$k" --dataset "$dataset" --push_embedding --template_portion loso
done