#!/bin/bash

#################### kNN ####################
# define following parameter
input_dir="hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/" #"hdfs://localhost:9000/user/rzhu9225/"
f_name="DecisionTree_final.py"
d=100
size=100
##############################################
echo "with two executor number 5 with two core numbers 2"
echo "check decision tree with pca dimension at 20 and 200"
echo "check multilayer perceptron with one hidden layer size at 50, 70 and 100"

for d in 200 20
do
	echo "n_exe=$n_exe, n_core=$n_core, d=$d"

cat > spark-submit.sh << EOF
spark-submit \
  --master yarn \
	--deploy-mode client \
  --num-executors 5 \
  --executor-cores 2 \
  DecisionTree_final.py \
  --input $input_dir \
  --d $d

EOF

				bash spark-submit.sh
done

#for size in 100 70 50
#do
#	echo "n_exe=$n_exe, n_core=$n_core, size=$size"

#cat > spark-submit.sh << EOF
#spark-submit \
#  --master yarn \
#	--deploy-mode client \
#  --num-executors 5 \
#  --executor-cores 2 \
#  MultilayerPerceptron_final.py \
#  --input $input_dir \
#  --size $size

#EOF

#				bash spark-submit.sh
#done
