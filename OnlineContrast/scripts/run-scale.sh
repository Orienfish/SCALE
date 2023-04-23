# Usage:
#   Run SCLL on ALL dataset under FIVE streaming settings: iid, seq, seq-bl, seq-cc, seq-im, e.g.,
#     ./run-scll.sh scale mnist iid trial#
#   Method choices: scale
#   Dataset choices: mnist, cifar10, cifar100, tinyimagenet
#   Data stream choices: iid, seq, seq-bl, seq-cc, seq-im
#   Trial #: the number of trial

cd ..;

lr=0.03
model=resnet18
distill_power=0.15;
mem_samples=128;
mem_size=256;
epochs=1;
if [ $3 = "iid" ]; then
  thres_ratio=0.3;
else  # seq, seq-bl, seq-cc, seq-im
  thres_ratio=0.1;
fi

if [ $2 = "mnist" ] || [ $2 = "svhn" ]; then
  for cluster_type in psa # maximin energy max_coverage kmeans
  do
    if [ $3 = "iid" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model cnn --training_data_type iid  \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
    fi

    if [ $3 = "seq" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model cnn --training_data_type class_iid  \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
    fi

    if [ $3 = "seq-bl" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model cnn --training_data_type class_iid --blend_ratio 0.5 \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
    fi

    if [ $3 = "seq-cc" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model cnn --training_data_type class_iid --n_concurrent_classes 2 \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
    fi

    if [ $3 = "seq-im" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model cnn --training_data_type class_iid \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 2048 4096 2048 4096 2048 2048 4096 2048 2048 4096 \
        --test_samples_per_cls 500 --knn_samples 1000 --trial $4
    fi
  done
fi


if [ $2 = "cifar10" ] ; then
  for cluster_type in psa # maximin energy max_coverage kmeans
  do
    if [ $3 = "iid" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type iid  \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
    fi

    if [ $3 = "seq" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid  \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
    fi

    if [ $3 = "seq-bl" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid --blend_ratio 0.5 \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
    fi

    if [ $3 = "seq-cc" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid --n_concurrent_classes 2 \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 4096 --test_samples_per_cls 500 --knn_samples 1000 --trial $4
    fi

    if [ $3 = "seq-im" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 2048 4096 2048 4096 2048 2048 4096 2048 2048 4096 \
      --test_samples_per_cls 500 --knn_samples 1000 --trial $4
    fi
  done
fi


if [ $2 = "cifar100" ] ; then
  for cluster_type in psa # maximin energy max_coverage kmeans
  do
    if [ $3 = "iid" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type iid  \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial $4
    fi

    if [ $3 = "seq" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid  \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial $4
    fi

    if [ $3 = "seq-bl" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid --blend_ratio 0.5 \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial $4
    fi

    if [ $3 = "seq-cc" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid --n_concurrent_classes 2 \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 2560 --test_samples_per_cls 250 --knn_samples 5000 --trial $4
    fi

    if [ $3 = "seq-im" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 1280 2560 2560 2560 1280 2560 2560 2560 2560 1280 2560 2560 2560 2560 1280 2560 2560 2560 2560 1280 \
        --test_samples_per_cls 250 --knn_samples 5000 --trial $4
    fi
  done
fi


if [ $2 = "tinyimagenet" ] ; then
  for cluster_type in psa # maximin energy max_coverage kmeans
  do
    if [ $3 = "iid" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type iid  \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples 100 --trial $4
    fi

    if [ $3 = "seq" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid  \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples 100 --trial $4
    fi

    if [ $3 = "seq-bl" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid --blend_ratio 0.5\
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples 100 --trial $4
    fi

    if [ $3 = "seq-cc" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid --n_concurrent_classes 2 \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 500 --test_samples_per_cls 50 --knn_samples 100 --trial $4
    fi

    if [ $3 = "seq-im" ]; then
      python main_supcon.py --criterion $1 --lifelong_method scale --dataset $2 --model resnet18 --training_data_type class_iid \
        --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs $epochs \
        --learning_rate_stream $lr --temp_cont 0.1 --simil tSNE --temp_tSNE 0.1 --thres_ratio $thres_ratio --distill_power $distill_power \
        --mem_update_type mo_rdn --mem_cluster_type $cluster_type --mem_max_new_ratio 0.1 --mem_max_classes 10 \
        --train_samples_per_cls 250 500 500 500 250 500 500 500 500 500 \
        --test_samples_per_cls 50 --knn_samples 100 --trial $4
    fi
  done
fi
