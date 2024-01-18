echo ">>> TRAINING melSeg WITH example_configs/training_config_08.yaml"
python3 melseg_trainer.py --config example_configs/training_config_08.yaml --cuda_id 2

echo "\n\n\n"

echo ">>> TRAINING melSeg WITH example_configs/training_config_06.yaml"
python3 melseg_trainer.py --config example_configs/training_config_06.yaml --cuda_id 2

echo "\n\n\n"

echo ">>> TRAINING melSeg WITH example_configs/training_config_04.yaml"
python3 melseg_trainer.py --config example_configs/training_config_04.yaml --cuda_id 2