#!/bin/bash

# read scenario from command line
if [ "$1" == "" ]; then
	scenario=final
else
	scenario=$1
fi

# set parameters according to chosen scenario
if [ "$scenario" == "submission" ]; then
        noiseSigma=0.01
        weights=0
        N=3000
elif [ "$scenario" == "poster" ]; then
        noiseSigma=0.1
        weights=0
        N=3000
elif [ "$scenario" == "posweights" ]; then
        noiseSigma=0.01
        weights=-2
        N=3000
elif [ "$scenario" == "posweights_noisy" ]; then
        noiseSigma=0.1
        weights=-2
        N=3000
elif [ "$scenario" == "lowN" ]; then
        noiseSigma=0.1
        weights=0
        N=100
elif [ "$scenario" == "final" ]; then
        noiseSigma=0.1
        weights=0
        N=3000
elif [ "$scenario" == "extY" ]; then
        noiseSigma=0.1
        weights=0
        N=3000
        # and manually set p to 6 or 7!
fi

bottomup=0
number=100
Nint=1000
mkdir plots
datname=exp_"$scenario".dat
fulldatname=plots/$datname
rm $fulldatname
for p in 10 30 50 70; do
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
done

gnuplotname=plots/exp_"$scenario".gnuplot
rm $gnuplotname
echo "set terminal pdf" >> $gnuplotname
echo "set output 'exp_""$scenario""_prediction.pdf'" >> $gnuplotname
echo "set xrange [0:100]" >> $gnuplotname
echo "set yrange [0:2]" >> $gnuplotname
echo "set xlabel 'p'" >> $gnuplotname
echo "set ylabel 'l1 error'" >> $gnuplotname
echo "plot '""$datname""' u 1:2 w linespoints title 'Yext' lt 1 pt 5, \\" >> $gnuplotname
echo "     '""$datname""' u 1:3 w linespoints title 'Y' lt 2 pt 7, \\" >> $gnuplotname
echo "     '""$datname""' u 1:4 w linespoints title 'Y1' lt 3 pt 9, \\" >> $gnuplotname
echo "     '""$datname""' u 1:5 w linespoints title 'Y2' lt 5 pt 13, \\" >> $gnuplotname
echo "     '""$datname""' u 1:6 w linespoints title 'baseline 1' lt 7 pt 1, \\" >> $gnuplotname
echo "     '""$datname""' u 1:7 w linespoints title 'baseline 2' lt 7 pt 2" >> $gnuplotname
echo >> $gnuplotname
echo "set output 'exp_""$scenario""_discovery.pdf'" >> $gnuplotname
echo "set xrange [0:100]" >> $gnuplotname
echo "set yrange [0:1]" >> $gnuplotname
echo "set xlabel 'p'" >> $gnuplotname
echo "set ylabel 'precision'" >> $gnuplotname
echo "plot '""$datname""' u 1:8 w linespoints title 'Yext' lt 1 pt 5, \\" >> $gnuplotname
echo "     '""$datname""' u 1:9 w linespoints title 'Y' lt 2 pt 7, \\" >> $gnuplotname
echo "     '""$datname""' u 1:10 w linespoints title 'Y1' lt 3 pt 9, \\" >> $gnuplotname
echo "     '""$datname""' u 1:11 w linespoints title 'Y2' lt 5 pt 13, \\" >> $gnuplotname
echo "     '""$datname""' u 1:12 w linespoints title 'baseline' lt 7 pt 4" >> $gnuplotname

cd plots
gnuplot exp_"$scenario".gnuplot
cd -
