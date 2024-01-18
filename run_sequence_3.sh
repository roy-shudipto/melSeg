echo ">>> TRAINING melSeg WITH example_configs/training_config_20.yaml"
python3 melseg_trainer.py --config example_configs/training_config_20.yaml --cuda_id 3

echo "\n\n\n"

echo ">>> TRAINING melSeg WITH example_configs/training_config_22.yaml"
python3 melseg_trainer.py --config example_configs/training_config_22.yaml --cuda_id 3

echo "\n\n\n"

echo ">>> TRAINING melSeg WITH example_configs/training_config_24.yaml"
python3 melseg_trainer.py --config example_configs/training_config_24.yaml --cuda_id 3