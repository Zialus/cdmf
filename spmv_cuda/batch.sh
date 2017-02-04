#!/bin/bash

# ----- dataset loop begin -----

DS=(jester ml-20m netflix yahoor1)
LWS=(1 2 4 8 16 32 64)

rawA="dat.native.u"

if [ -e ${rawA} ]
then
        rm ${rawA}
fi

echo "cpu:" >> ${rawA} 
for ds in ${DS[@]}; do
        for lws in ${LWS[@]}; do
                ./spmv -P 0 -nThreadsPerBlock $lws ../datasetcf/${ds} > tmp.dat
                echo -ne `grep "spmv time" ./tmp.dat | cut -d':' -f2` >> ${rawA}
                echo -ne "\t" >> ${rawA}
        done
        echo "" >> ${rawA}
done

echo "gpu:" >> ${rawA} 
for ds in ${DS[@]}; do
        for lws in ${LWS[@]}; do
                ./spmv -P 1 -nThreadsPerBlock $lws ../datasetcf/${ds} > tmp.dat
                echo -ne `grep "spmv time" ./tmp.dat | cut -d':' -f2` >> ${rawA}
                echo -ne "\t" >> ${rawA}
        done
        echo "" >> ${rawA}
done

echo "mic:" >> ${rawA} 
for ds in ${DS[@]}; do
        for lws in ${LWS[@]}; do
                ./spmv -P 2 -nThreadsPerBlock $lws ../datasetcf/${ds} > tmp.dat
                echo -ne `grep "spmv time" ./tmp.dat | cut -d':' -f2` >> ${rawA}
                echo -ne "\t" >> ${rawA}
        done
        echo "" >> ${rawA}
done

