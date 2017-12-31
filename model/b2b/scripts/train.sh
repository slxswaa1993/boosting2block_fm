
debug=$1
iters=$2
eta=$3
alpha_1=$4
alpha_3=$5
lepoc=$6
save=$7
DATE=$(date +%m%d%H%M)
nohup python -u ../main.py -debug ${debug} -iters ${iters} -lepoc ${lepoc} -eta ${eta} -alpha1 ${alpha_1} -alpha3 ${alpha_3} -save ${save}> ./trian_log/train_${debug}_${DATE}.log 2>&1 &




