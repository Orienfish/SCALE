for trial in 0 1 2
do
  for dataset in cifar10 # cifar100 tinyimagenet
  do
    for datatype in seq-im # iid seq seq-bl seq-cc
    do
      # the default loss is simclr
      bash ./run-co2l.sh simclr $dataset $datatype $trial
    done
  done
done