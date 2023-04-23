for trial in 0 1 2
do
  for dataset in cifar10 # cifar100 tinyimagenet
  do
    for baseline in simclr
    do
      for datatype in seq-im # seq iid seq-bl seq-cc
      do
        (( bash ./run-simclr.sh $baseline $dataset $datatype $trial ) 2>&1 ) | tee log_"$baseline"_"$dataset"_"$datatype"_"$trial"
      done
    done
  done
done
