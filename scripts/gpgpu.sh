#!/bin/bash
set -x

#DS=(jester ml-20m netflix yahoor1)
DS=(jester)
#LWS=(1 2 3 4 5 6 7 8 9 10)
LWS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
TS=(1 3 5 7)


#-----------------------   V2   -----------------------#
rawA="dat.float.rmse.v2"

if [[ -e ${rawA} ]]
then
    rm ${rawA}
fi

for ts in "${TS[@]}"; do
    echo "T=${ts}:" >> ${rawA}
    for ds in "${DS[@]}"; do
        for lws in "${LWS[@]}"; do
            ../exec/cdmf -k 40 -t "$lws" -T "$ts" -P 0 -d 1 -l 0.05 -nThreadsPerBlock 64 -p 1 -V 2 ../data/"$ds"/ > tmp.dat
            echo -ne $(grep "OCL Training time" ./tmp.dat | cut -d':' -f2 | cut -d' ' -f2) >> ${rawA}
            echo -ne "\\t" >> ${rawA}
            echo -ne $(grep "Test RMSE" ./tmp.dat | cut -d'=' -f2 | cut -d' ' -f2) >> ${rawA}
            echo "" >> ${rawA}
        done
        echo "" >> ${rawA}
    done
done



#-----------------------   V1   -----------------------#
rawB="dat.float.rmse.v1"

if [[ -e ${rawB} ]]
then
    rm ${rawB}
fi

for ts in "${TS[@]}"; do
    echo "T=${ts}:" >> ${rawB}
    for ds in "${DS[@]}"; do
        for lws in "${LWS[@]}"; do
             ../exec/cdmf -k 40 -t "$lws" -T "$ts" -P 0 -d 1 -l 0.05 -nThreadsPerBlock 64 -p 1 -V 1 ../data/"$ds"/ > tmp.dat
             echo -ne $(grep "OCL Training time" ./tmp.dat | cut -d':' -f2 | cut -d' ' -f2) >> ${rawB}
             echo -ne "\\t" >> ${rawB}
             echo -ne $(grep "Test RMSE" ./tmp.dat | cut -d'=' -f2 | cut -d' ' -f2) >> ${rawB}
             echo "" >> ${rawB}
        done
        echo "" >> ${rawB}
    done
done
