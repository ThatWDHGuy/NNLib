CC=g++
CFLAGS=-I.
DEPS = neuron.h nnlib.h
OBJ = main.o neuron.o nnlib.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean: $(OBJ)
	rm *.o