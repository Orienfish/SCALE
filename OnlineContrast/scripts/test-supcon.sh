for trial in 0
do
  for dataset in cifar10 cifar100
  do
    for datatype in iid seq seq-bl seq-cc seq-im
    do
      bash ./run-supcon.sh supcon $dataset $datatype $trial
    done
  done
done