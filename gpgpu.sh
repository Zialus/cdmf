#!/bin/bash

# ----- dataset loop begin -----

#DS=(jester ml-20m netflix yahoor1)
DS=(netflix)
#LWS=(1 2 3 4 5 6 7 8 9 10)
LWS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

rawA="dat.float.rmse"

#if [ -e ${rawA} ]
#then
#        rm ${rawA}
#fi

#echo "T=1:" >> ${rawA} 
#for ds in ${DS[@]}; do
#        for lws in ${LWS[@]}; do
#               #./cdmf -k 40 -t 5 -T 5 -n 24 -l 0.05 -nThreadsPerBlock 512 -p 1 ./data/netflix/
#               ./cdmf -k 40 -t ${lws} -T 1 -P 0 -l 0.05 -nThreadsPerBlock 512 -p 1 ./data/${ds}/ > tmp.dat
#                #./cdmf -P 0 -nThreadsPerBlock $lws ../datasetcf/${ds} > tmp.dat
#                echo -ne `grep "training time" ./tmp.dat | cut -d':' -f2 | cut -d' ' -f2` >> ${rawA}
#                echo -ne "\t" >> ${rawA}
#                echo -ne `grep "test RMSE" ./tmp.dat | cut -d'=' -f2 | cut -d' ' -f2` >> ${rawA}
#                #echo -ne "\t" >> ${rawA}
#               echo "" >> ${rawA}
#        done
#        echo "" >> ${rawA}
#done

#echo "T=3:" >> ${rawA} 
#for ds in ${DS[@]}; do
#        for lws in ${LWS[@]}; do
#               #./cdmf -k 40 -t 5 -T 5 -n 24 -l 0.05 -nThreadsPerBlock 512 -p 1 ./data/netflix/
#               ./cdmf -k 40 -t ${lws} -T 3 -P 0 -l 0.05 -nThreadsPerBlock 512 -p 1 ./data/${ds}/ > tmp.dat
#                #./cdmf -P 0 -nThreadsPerBlock $lws ../datasetcf/${ds} > tmp.dat
#                echo -ne `grep "training time" ./tmp.dat | cut -d':' -f2 | cut -d' ' -f2` >> ${rawA}
#                echo -ne "\t" >> ${rawA}
#                echo -ne `grep "test RMSE" ./tmp.dat | cut -d'=' -f2 | cut -d' ' -f2` >> ${rawA}
#                #echo -ne "\t" >> ${rawA}
#               echo "" >> ${rawA}
#        done
#        echo "" >> ${rawA}
#done

echo "T=5:" >> ${rawA} 
for ds in ${DS[@]}; do
        for lws in ${LWS[@]}; do
                #./cdmf -k 40 -t 5 -T 5 -n 24 -l 0.05 -nThreadsPerBlock 512 -p 1 ./data/netflix/
                ./cdmf -k 40 -t ${lws} -T 5 -P 0 -l 0.05 -nThreadsPerBlock 512 -p 1 ./data/${ds}/ > tmp.dat
                #./cdmf -P 0 -nThreadsPerBlock $lws ../datasetcf/${ds} > tmp.dat
                echo -ne `grep "training time" ./tmp.dat | cut -d':' -f2 | cut -d' ' -f2` >> ${rawA}
                echo -ne "\t" >> ${rawA}
                echo -ne `grep "test RMSE" ./tmp.dat | cut -d'=' -f2 | cut -d' ' -f2` >> ${rawA}
                #echo -ne "\t" >> ${rawA}
                echo "" >> ${rawA}
        done
        echo "" >> ${rawA}
done

echo "T=7:" >> ${rawA} 
for ds in ${DS[@]}; do
        for lws in ${LWS[@]}; do
                #./cdmf -k 40 -t 5 -T 5 -n 24 -l 0.05 -nThreadsPerBlock 512 -p 1 ./data/netflix/
                ./cdmf -k 40 -t ${lws} -T 7 -P 0 -l 0.05 -nThreadsPerBlock 512 -p 1 ./data/${ds}/ > tmp.dat
                #./cdmf -P 0 -nThreadsPerBlock $lws ../datasetcf/${ds} > tmp.dat
                echo -ne `grep "training time" ./tmp.dat | cut -d':' -f2 | cut -d' ' -f2` >> ${rawA}
                echo -ne "\t" >> ${rawA}
                echo -ne `grep "test RMSE" ./tmp.dat | cut -d'=' -f2 | cut -d' ' -f2` >> ${rawA}
                #echo -ne "\t" >> ${rawA}
                echo "" >> ${rawA}
        done
        echo "" >> ${rawA}
done

#-------------------------------------------------
rawB="dat.float.rmse.time"

if [ -e ${rawB} ]
then
        rm ${rawB}
fi

echo "T=1:" >> ${rawB} 
for ds in ${DS[@]}; do
        for lws in ${LWS[@]}; do
                ./cdmf.v3 -k 40 -t ${lws} -T 1 -P 0 -l 0.05 -nThreadsPerBlock 512 ./data/${ds}/ > tmp.dat
                echo -ne `grep "training time" ./tmp.dat | cut -d':' -f2 | cut -d' ' -f2` >> ${rawB}
                echo "" >> ${rawB}
        done
        echo "" >> ${rawB}
done

echo "T=3:" >> ${rawB} 
for ds in ${DS[@]}; do
        for lws in ${LWS[@]}; do
                ./cdmf.v3 -k 40 -t ${lws} -T 3 -P 0 -l 0.05 -nThreadsPerBlock 512 ./data/${ds}/ > tmp.dat
                echo -ne `grep "training time" ./tmp.dat | cut -d':' -f2 | cut -d' ' -f2` >> ${rawB}
                echo "" >> ${rawB}
        done
        echo "" >> ${rawB}
done

echo "T=5:" >> ${rawB} 
for ds in ${DS[@]}; do
        for lws in ${LWS[@]}; do
                ./cdmf.v3 -k 40 -t ${lws} -T 5 -P 0 -l 0.05 -nThreadsPerBlock 512 ./data/${ds}/ > tmp.dat
                echo -ne `grep "training time" ./tmp.dat | cut -d':' -f2 | cut -d' ' -f2` >> ${rawB}
                echo "" >> ${rawB}
        done
        echo "" >> ${rawB}
done

echo "T=7:" >> ${rawB} 
for ds in ${DS[@]}; do
        for lws in ${LWS[@]}; do
                ./cdmf.v3 -k 40 -t ${lws} -T 7 -P 0 -l 0.05 -nThreadsPerBlock 512 ./data/${ds}/ > tmp.dat
                echo -ne `grep "training time" ./tmp.dat | cut -d':' -f2 | cut -d' ' -f2` >> ${rawB}
                echo "" >> ${rawB}
        done
        echo "" >> ${rawB}
done
