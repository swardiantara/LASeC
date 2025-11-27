
initial_models=( all-MiniLM-L6-v2 )
samplings=( random )
num_samples=( 2 1 )


for model in "${initial_models[@]}"; do
    for k in "${num_samples[@]}"; do
        for sampling in "${samplings[@]}"; do
            python -m src.train_embedding --initial_model_path "$model" --sampling_strategy "$sampling" --k "$k" --push_embedding --template_portion partial
        done
    done
done