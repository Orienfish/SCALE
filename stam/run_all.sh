for dataset in cifar10 cifar100 mnist
do
    for datatype in iid seq seq-bl seq-cc seq-im
    do
        bash ./stam_IJCAI.sh $dataset $datatype
    done
done
