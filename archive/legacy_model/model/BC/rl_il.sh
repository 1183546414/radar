python offline_rl_il_from_json.py \
  --encoded_jamming_type /mnt/data/Encoded_Jamming_Type.json \
  --jammer_carrier_frequency /mnt/data/Jammer_carrierFrequency.json \
  --jamming_params /mnt/data/Jamming_Params.json \
  --radar_carrier_frequency /mnt/data/Radar_carrierFrequency.json \
  --radar_params /mnt/data/Radar_Params.json \
  --sinr /mnt/data/SINR.json \
  --epochs_bc 100 --epochs_joint 200 --lambda_rl 0.7