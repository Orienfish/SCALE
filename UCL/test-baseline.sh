for trial in 0 1 2
do
  for dataset in cifar10 cifar100 tinyimagenet
  do
    for datatype in seq-im seq iid seq-bl seq-cc
    do
      for method in si pnn der mixup
      do
      for backbone in simclr
        do
          (( bash ./run-baseline.sh $method $backbone $dataset $datatype $trial ) 2>&1 ) | tee log_"$method"_"$backbone"_"$dataset"_"$datatype"_"$trial"
        done
      done
    done
  done
done
