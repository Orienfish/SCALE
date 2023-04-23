for trial in 0
do
  for dataset in cifar10 # cifar100
  do
    for datatype in seq-im # iid seq seq-bl seq-cc
    do
      bash ./run-supcon.sh supcon $dataset $datatype $trial
    done
  done
done