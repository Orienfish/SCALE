### Incremental UPL
if [ "$1" = "mnist" ]; then
  if [ "$2" = "iid" ]; then
    python3 main.py --model_name stam --dataset mnist --N_l 10 --N_e 500 --N_p 10000 --ntrials 3 --ntp 1 --log stam_iid --load_log stam_iid --schedule_flag 2 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400 --k_scale 1.0 --training_data_type iid --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq" ]; then
    python3 main.py --model_name stam --dataset mnist --N_l 10 --N_e 500 --N_p 10000 --ntrials 3 --ntp 1 --log stam_seq --load_log stam_seq --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400 --k_scale 1.0 --training_data_type sequential --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq-bl" ]; then
    python3 main.py --model_name stam --dataset mnist --N_l 10 --N_e 500 --N_p 10000 --ntrials 3 --ntp 1 --log stam_seq_blend --load_log stam_seq_blend --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400 --k_scale 1.0 --training_data_type sequential --blend_ratio 0.5 --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq-cc" ]; then
    python3 main.py --model_name stam --dataset mnist --N_l 10 --N_e 500 --N_p 10000 --ntrials 3 --ntp 1 --log stam_seq_concurrent --load_log stam_seq_concurrent --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400 --k_scale 1.0 --training_data_type sequential --n_concurrent_classes 2 --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq-im" ]; then
    python3 main.py --model_name stam --dataset mnist --N_l 10 --N_e 500 --N_p 10000 --ntrials 3 --ntp 1 --log stam_seq_concurrent --load_log stam_seq_concurrent --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400 --k_scale 1.0 --training_data_type sequential --train_samples_per_cls 2048 4096 2048 4096 2048 2048 4096 2048 2048 4096
  fi
fi

if [ "$1" = "svhn" ]; then
  if [ "$2" = "iid" ]; then
    python3 main.py --model_name stam --dataset svhn --N_l 50 --N_e 500 --N_p 10000 --ntrials 2 --ntp 1 --log stam_seq_concurrent --load_log stam_seq_concurrent --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 2000 --k_scale 1.0 --training_data_type iid --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq" ]; then
    python3 main.py --model_name stam --dataset svhn --N_l 50 --N_e 500 --N_p 10000 --ntrials 2 --ntp 1 --log stam_seq_concurrent --load_log stam_seq_concurrent --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 2000 --k_scale 1.0 --training_data_type sequential --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq-bl" ]; then
    python3 main.py --model_name stam --dataset svhn --N_l 50 --N_e 500 --N_p 10000 --ntrials 2 --ntp 1 --log stam_seq_concurrent --load_log stam_seq_concurrent --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 2000 --k_scale 1.0 --training_data_type sequential --blend_ratio 0.5 --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq-cc" ]; then
    python3 main.py --model_name stam --dataset svhn --N_l 50 --N_e 500 --N_p 10000 --ntrials 2 --ntp 1 --log stam_seq_concurrent --load_log stam_seq_concurrent --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 2000 --k_scale 1.0 --training_data_type sequential --n_concurrent_classes 2 --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq-im" ]; then
    python3 main.py --model_name stam --dataset svhn --N_l 50 --N_e 500 --N_p 10000 --ntrials 2 --ntp 1 --log stam_seq_concurrent --load_log stam_seq_concurrent --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 2000 --k_scale 1.0 --training_data_type sequential --train_samples_per_cls 2048 4096 2048 4096 2048 2048 4096 2048 2048 4096
  fi
fi


if [ "$1" = "cifar10" ]; then
  if [ "$2" = "iid" ]; then
    python3 main.py --model_name stam --dataset cifar10 --N_l 100 --N_e 1000 --N_p 10000 --ntrials 3 --ntp 1 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb --delta 2500 --k_scale 1.0 --training_data_type iid --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq" ]; then
    python3 main.py --model_name stam --dataset cifar10 --N_l 100 --N_e 1000 --N_p 10000 --ntrials 3 --ntp 1 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb --delta 2500 --k_scale 1.0 --training_data_type sequential --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq-bl" ]; then
    python3 main.py --model_name stam --dataset cifar10 --N_l 100 --N_e 1000 --N_p 10000 --ntrials 3 --ntp 1 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb --delta 2500 --k_scale 1.0 --training_data_type sequential --blend_ratio 0.5 --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq-cc" ]; then
    python3 main.py --model_name stam --dataset cifar10 --N_l 100 --N_e 1000 --N_p 10000 --ntrials 3 --ntp 1 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb --delta 2500 --k_scale 1.0 --training_data_type sequential --n_concurrent_classes 2 --train_samples_per_cls 4096
  fi

  if [ "$2" = "seq-im" ]; then
    python3 main.py --model_name stam --dataset cifar10 --N_l 100 --N_e 1000 --N_p 10000 --ntrials 3 --ntp 1 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb --delta 2500 --k_scale 1.0 --training_data_type sequential --train_samples_per_cls 2048 4096 2048 4096 2048 2048 4096 2048 2048 4096
  fi
fi

if [ "$1" = "cifar100" ]; then
  if [ "$2" = "iid" ]; then
    python3 main.py --model_name stam --dataset cifar100 --N_l 100 --N_e 1000 --N_p 10000 --ntrials 3 --ntp 1 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb --delta 2500 --k_scale 1.0 --training_data_type iid --train_samples_per_cls 2560
  fi

  if [ "$2" = "seq" ]; then
    python3 main.py --model_name stam --dataset cifar100 --N_l 100 --N_e 1000 --N_p 10000 --ntrials 3 --ntp 1 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb --delta 2500 --k_scale 1.0 --training_data_type sequential --train_samples_per_cls 2560
  fi

  if [ "$2" = "seq-bl" ]; then
    python3 main.py --model_name stam --dataset cifar100 --N_l 100 --N_e 1000 --N_p 10000 --ntrials 3 --ntp 1 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb --delta 2500 --k_scale 1.0 --training_data_type sequential --blend_ratio 0.5 --train_samples_per_cls 2560
  fi

  if [ "$2" = "seq-cc" ]; then
    python3 main.py --model_name stam --dataset cifar100 --N_l 100 --N_e 1000 --N_p 10000 --ntrials 3 --ntp 1 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb --delta 2500 --k_scale 1.0 --training_data_type sequential --n_concurrent_classes 2 --train_samples_per_cls 2560
  fi

  if [ "$2" = "seq-im" ]; then
    python3 main.py --model_name stam --dataset cifar100 --N_l 100 --N_e 1000 --N_p 10000 --ntrials 3 --ntp 1 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb --delta 2500 --k_scale 1.0 --training_data_type sequential --train_samples_per_cls 1280 2560 2560 2560 1280 2560 2560 2560 2560 1280 2560 2560 2560 2560 1280 2560 2560 2560 2560 1280
  fi
fi

#conpython3 main.py --model_name stam --dataset emnist --N_l 10 --N_e 100 --N_p 2000 --ntrials 3 --ntp 5 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400


### Uniform UPL
#python3 main.py --model_name stam --dataset mnist --N_l 10 --N_e 100 --N_p 10000 --ntrials 3 --ntp 5 --log stam_uniform --load_log stam_uniform --schedule_flag 2 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400 

#python3 main.py --model_name stam --dataset svhn --N_l 100 --N_e 100 --N_p 10000 --ntrials 3 --ntp 5 --log stam_uniform --load_log stam_uniform --schedule_flag 2 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 2000

#python3 main.py --model_name stam --dataset cifar-10 --N_l 100 --N_e 100 --N_p 10000 --ntrials 3 --ntp 5 --log stam_uniform --load_log stam_uniform --schedule_flag 2 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb  --delta 2500

#python3 main.py --model_name stam --dataset emnist --N_l 10 --N_e 100 --N_p 2000 --ntrials 3 --ntp 5 --log stam_uniform --load_log stam_uniform --schedule_flag 2 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400
