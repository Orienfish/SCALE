for trial in 0 1 2
do
  for dataset in cifar10 cifar100 tinyimagenet
  do
    for datatype in seq-im seq iid seq-bl seq-cc
    do
      for criterion in simclr
      do
        (( bash ./run-cassle.sh $criterion $dataset $datatype $trial ) 2>&1 ) | tee log_cassle_"$criterion"_"$dataset"_"$datatype"_"$trial"
      done
    done
  done
done
