CC=g++
CFLAGS=-c -O2 -fpic -std=c++17 -march=x86-64-v3
LDFLAGS=-Ltensorflow-cc
INCL=-I/usr/include/tensorflow -I narf/include

narf/lib/libnarf.so : narf/lib/tfccutils.o
	$(CC) $(LDFLAGS) -shared -o narf/lib/libnarf.so narf/lib/tfccutils.o

narf/lib/tfccutils.o : narf/src/tfccutils.cpp narf/include/tfccutils.h
	$(CC) $(CFLAGS) $(INCL) -o narf/lib/tfccutils.o narf/src/tfccutils.cpp
