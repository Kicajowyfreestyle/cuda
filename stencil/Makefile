objects = filter.o filterArray.o

all: filter clean

filter: $(objects)
		nvcc $(objects) -o filter

%.o: %.cpp
		nvcc -x cu -dc $< -o $@

clean:
		rm -f *.o
