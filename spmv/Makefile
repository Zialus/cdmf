CXX=g++
CXXFLAGS=-fopenmp -lOpenCL -O3 -Wno-format-security -Wno-unused-result 
DFLAGS=-DNUM_RUN=100
VERSION=1.41
INC=/home/jianbin/ocl/include/
EXE=spmv
#all: omp-pmf-train omp-pmf-predict
all: $(EXE)
db: ocl-pmf-train-db 

$(EXE): $(EXE).cpp util.cpp
	${CXX}  ${DFLAGS} -I${INC} -o $@ $^ ${CXXFLAGS}

#tar: 
#	make clean; cd ../;  tar cvzf libpmf-${VERSION}.tgz libpmf-${VERSION}/

clean:
#	make -C python clean
#	make -C R clean
#	make -C matlab
	rm -rf $(EXE)  *.o *~

