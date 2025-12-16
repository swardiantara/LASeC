#!/bin/bash
python -m src.train_embedding --initial_model_path all-MiniLM-L6-v2 --sampling_strategy random --k 1 --push_embedding --batch_size 128