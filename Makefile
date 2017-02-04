CXX=g++
CXXFLAGS=-fopenmp -lOpenCL -O3 -Wno-format-security -Wno-unused-result -Wno-write-strings
DFLAGS=-DNUM_RUN=1
VERSION=V1	# V1: native, V2: thread batching, V3: load balancing
INC=/home/jianbin/ocl/include/
EXE=cdmf
SRC=$(EXE).cpp $(EXE)_ref.cpp $(EXE)_ocl.cpp $(EXE)_csr5.cpp $(EXE)_native.cpp util.cpp
all: $(EXE)
db: $(EXE).db

$(EXE): $(SRC)
	${CXX} ${DFLAGS} -I${INC} -o $@ $^ ${CXXFLAGS} -D${VERSION}
$(EXE).db: $(SRC)
	${CXX} ${DFLAGS} -I${INC} -o $@ $^ ${CXXFLAGS} -DDB_INFO

#tar: 
#	make clean; cd ../;  tar cvzf libpmf-${VERSION}.tgz libpmf-${VERSION}/

clean:
	rm -rf $(EXE) $(EXE) $(EXE).db  *.o *~ ./kcode/*~

