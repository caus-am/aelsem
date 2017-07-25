#!/bin/bash

# read p from command line
if [ "$1" == "" ]; then
    p=10
else
    p=$1
fi

if [ "$p" == "6" ] || [ "$p" == "7" ]; then
    scenario=extY
else
    scenario=final
fi

# set parameters according scenario "final" for run_experiment.sh
noiseSigma=0.1
weights=0
N=3000

bottomup=0
number=100
Nint=1000
mkdir plots
datname=exp_"$scenario".dat
fulldatname=plots/$datname
rm $fulldatname

expname=exp_p"$p"_$scenario

# simulate data
Rscript experiment_simulate.R $expname $p $number $bottomup $N $Nint $noiseSigma $weights

# analyze data (each DAG individually)
#rm $expname/simulData.csv
#for i in $expname/simulData*.Rdata; do
#        echo Analyzing $i
#        ipython experiment_analyse.py /tmp/simulData.csv /tmp/simulData.tex $i 
#       cat /tmp/simulData.csv >> $expname/simulData.csv
#done

# analyze data (all DAGs together)
echo Analyzing $expname/simulData*.Rdata
time ipython experiment_analyse.py $expname/simulData.csv $expname/simulData.tex $expname/simulData*.Rdata

# pdflatex the resulting file
cd $expname
pdflatex simulData.tex
cd -

# collect statistics for plot
echo -n "$p " >> $fulldatname
awk '{print $105,$111,$117,$123,$129,$131,$86,$92,$98,$104,($81+$84)/($81+$82+$83+$84)}' < $expname/simulData.csv >> $fulldatname 

# skip the gnuplot generation from run_experiment.sh
