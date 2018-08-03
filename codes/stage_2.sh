#!/bin/bash

#################### kNN ####################
# define following parameter
input_dir="hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/" #"hdfs://localhost:9000/user/rzhu9225/"
master="yarn"
deploy="client"
n_exe=5
n_core=4
f_name="kNN_Implementation_final.py"
d=100
k=5
##############################################
echo "check two executor number (5 & 4) with two core numbers (1 & 2)"
echo "check k valued at two 5 and 10 with pca dimension at 50 and 100"

for n_exe in 5 4
do
	for n_core in 1 2
	do
		for d in 80 50
		do
			for k in 10 5
			do
				echo "n_exe=$n_exe, n_core=$n_core, d=$d, k=$k"

cat > spark-submit.sh << EOF
spark-submit \
  --master yarn \
	--deploy-mode client \
  --num-executors $n_exe \
  --executor-cores $n_core \
  kNN_Implementation_final.py \
  --input $input_dir \
  --d $d \
  --k $k
EOF

				bash spark-submit.sh
			done
		done
	done
done
