objects = map.o mapArray.o

all: map clean

map: $(objects)
		nvcc $(objects) -o map

%.o: %.cpp
		nvcc -x cu -dc $< -o $@

clean:
		rm -f *.o
# all: mapArray
# mapArray: mapArray.o map.o
# 	$(NVCC) -o mapArray mapArray.o map.o
# mapArray.o: mapArray.cpp map.cu
# 	$(NVCC) -c -o mapArray.cpp map.cu
# map.o: map.cu map.cuh
# 	$(NVCC) -c -o map.o map.cu map.cuh

# mapArray.o: mapArray.cpp
# 	$(CC) -c %< -o $@
# map.o: map.cu map.cuh
# 	$(NVCC) -c %< -o $@
# mapArray: mapArray.o map.o
# 	$(NVCC) %^ -o $@

# all: mapArray

# mapArray: map.o mapArray.o
# 	$(NVCC) map.o mapArray.o -o mapArray

# mapArray.o: mapArray.cpp
# 	$(CC) -c mapArray.cpp

# map.o: map.cu
# 	$(NVCC) -c map.cu

#nvcc map.cu mapArray.cpp -o map
