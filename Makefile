CC=g++
CXXFLAGS=-std=c++17 -I. -Wall -Wno-sign-compare
DEPS = neuron.h nnlib.h trainitem.h
OBJ = main.o neuron.o nnlib.o trainitem.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) 

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean: $(OBJ)
	rm *.o