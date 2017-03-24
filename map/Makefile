objects = map.o mapArray.o

all: map clean

map: $(objects)
		nvcc $(objects) -o map

%.o: %.cpp
		nvcc -x cu -dc $< -o $@

clean:
		rm -f *.o
