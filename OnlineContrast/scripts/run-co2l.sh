# Usage:
#   Run SCLL on ALL dataset under FIVE streaming settings: iid, seq, seq-bl, seq-cc, seq-im, e.g.,
#     ./run-scll.sh simclr mnist iid trial# ckpt
#   Criterion choices: simclr
#   Dataset choices: mnist, svhn, cifar10, cifar100, tinyimagenet
#   Data stream choices: iid, seq, seq-bl, seq-cc, seq-im
#   Trial #: the number of trial
#   ckpt: path to the stored model

cd ..;

lr=0.01;
model=resnet18;
mem_samples=128;
mem_size=256;
distill_power=0.15;

if [ $2 = "mnist" ] || [ $2 = "svhn" ]; then
  if [ $3 = "iid" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model cnn --training_data_type iid  \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 \
        --mem_update_type rdn --mem_max_classes 10 \
        --distill_power $distill_power --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
  fi

  if [ $3 = "seq" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model cnn --training_data_type class_iid \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 \
        --mem_update_type rdn --mem_max_classes 10 \
        --distill_power $distill_power --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
  fi

  if [ $3 = "seq-bl" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model cnn --training_data_type class_iid --blend_ratio 0.5 \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 \
        --mem_update_type rdn --mem_max_classes 10 \
        --distill_power $distill_power --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
  fi

  if [ $3 = "seq-cc" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model cnn --training_data_type class_iid --n_concurrent_classes 2 \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 \
        --mem_update_type rdn --mem_max_classes 10 \
        --distill_power $distill_power --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
  fi

  if [ $3 = "seq-im" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model cnn --training_data_type class_iid \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 \
        --mem_update_type rdn --mem_max_classes 10 \
        --distill_power $distill_power --train_samples_per_cls 2048 4096 2048 4096 2048 2048 4096 2048 2048 4096 \
        --test_samples_per_cls 500 --knn_samples 1000 --trial $4
  fi
fi


if [ $2 = "cifar10" ] ; then
  if [ $3 = "iid" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type iid  \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
  fi

  if [ $3 = "seq" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
  fi

  if [ $3 = "seq-bl" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
  fi

  if [ $3 = "seq-cc" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --mem_samples 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
  fi

  if [ $3 = "seq-im" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 2048 4096 2048 4096 2048 2048 4096 2048 2048 4096 \
      --test_samples_per_cls 500 --knn_samples 1000 --trial $4
  fi
fi


if [ $2 = "cifar100" ] ; then
  if [ $3 = "iid" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type iid  \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial $4
  fi

  if [ $3 = "seq" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial $4
  fi

  if [ $3 = "seq-bl" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --mem_samples 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial $4
  fi

  if [ $3 = "seq-cc" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial $4
  fi

  if [ $3 = "seq-im" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 1280 2560 2560 2560 1280 2560 2560 2560 2560 2560 2560 2560 2560 2560 1280 2560 2560 2560 2560 1280 \
      --test_samples_per_cls 250 --knn_samples 5000 --trial $4
  fi
fi


if [ $2 = "tinyimagenet" ] ; then
  if [ $3 = "iid" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type iid  \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-bl" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-cc" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-im" ]; then
    python main_supcon.py --criterion $1 --lifelong_method co2l --dataset $2 --model $model --training_data_type class_iid \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --current_temp 0.1 \
      --mem_update_type rdn --mem_max_classes 10 \
      --distill_power $distill_power \
      --train_samples_per_cls 250 500 500 500 250 500 500 500 500 500 \
      --test_samples_per_cls 50 --knn_samples_ratio 1.0 --trial $4
  fi
fi