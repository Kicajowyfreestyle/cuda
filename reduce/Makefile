objects = reduce.o reduceArray.o

all: reduce clean

reduce: $(objects)
		nvcc $(objects) -o reduce

%.o: %.cpp
		nvcc -x cu -dc $< -o $@

clean:
		rm -f *.o
