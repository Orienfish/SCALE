# Usage:
#   Run UCL on various dataset under four streaming settings: iid, seq, seq-bl, seq-cc, e.g.,
#     ./run-baseline.sh mixup simclr mnist iid trial#
#   Method choices: mixup, pnn, si, der
#   Model name choices: simclr, simsiam, barlowtwins, byol, vicreg
#   Dataset choices: mnist, svhn, cifar10, cifar100, tinyimagenet
#   Datatype choices: iid, seq, seq-bl, seq-cc, seq-im
#   Trial #: the number of trial

model=resnet18
lr=0.03;

if [ "$3" = "mnist" ] || [ "$3" = "svhn" ]; then
  if [ "$4" = "iid" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone cnn --training_data_type iid  \
      --batch_size 256 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone cnn --training_data_type class_iid \
      --batch_size 256 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-bl" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone cnn --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 256 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-cc" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone cnn --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 256 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-im" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone cnn --training_data_type class_iid \
      --batch_size 256 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 2048 4096 2048 4096 2048 2048 4096 2048 2048 4096 \
      --test_samples_per_cls 500 --knn_samples 1000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi
fi

if [ "$3" = "cifar10" ] ; then
  if [ "$4" = "iid" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type iid \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid\
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-bl" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-cc" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-im" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 2048 4096 2048 4096 2048 2048 4096 2048 2048 4096 \
      --test_samples_per_cls 500 --knn_samples 1000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi
fi

if [ "$3" = "cifar100" ] ; then
  if [ "$4" = "iid" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type iid  \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --val_freq 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-bl" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-cc" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-im" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 1280 2560 2560 2560 1280 2560 2560 2560 2560 1280 2560 2560 2560 2560 1280 2560 2560 2560 2560 1280 \
      --test_samples_per_cls 250 --knn_samples 5000 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi
fi

if [ "$3" = "tinyimagenet" ] ; then
  if [ "$4" = "iid" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type iid  \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples 100 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples 100 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-bl" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples 100 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-cc" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples 100 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-im" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_per_cls 250 500 500 500 250 500 500 500 500 500 \
      --test_samples_per_cls 50 --knn_samples 100 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi
fi

