CXX      := g++
CXXFLAGS := -std=c++17 -O2
PKG      := $(shell pkg-config --cflags --libs opencv4)

SRCS := $(wildcard q*/main.cpp)
BINS := $(patsubst q%/main.cpp,q%/a.out,$(SRCS))

all: $(BINS)

q%/a.out: q%/main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(PKG)

clean:
	rm -f $(BINS)

.PHONY: all clean
