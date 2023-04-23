for trial in 0 1 2
do
  for dataset in cifar10 # cifar100 tinyimagenet
  do
    for datatype in seq-im # seq iid seq-bl seq-cc
    do
      bash ./run-scale.sh supcon $dataset $datatype $trial
    done
  done
done