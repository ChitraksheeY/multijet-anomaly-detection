#################################################################
# Compiler setting
CPPFLAGS += -Wall -g -fPIC -std=c++11 -O2 -fopenmp
LDFLAGS += -fopenmp
# ROOT
CPPFLAGS += $(shell root-config --cflags)
LDFLAGS += $(shell root-config --glibs) 

# FASTJET
CPPFLAGS += $(shell fastjet-config --cxxflags)
LDFLAGS += $(shell fastjet-config --libs) -lNsubjettiness -lEnergyCorrelator -lRecursiveTools -lfastjettools

#########################################################################
# sources: every src/*.cxx is built into ./<name>.run
EXE_DIR = $(PWD)/src
EXE_SRC = $(wildcard $(EXE_DIR)/*.cxx)
EXE_OBJ = $(EXE_SRC:$(EXE_DIR)/%.cxx=$(EXE_DIR)/%.o)
EXE_RUN = $(EXE_SRC:$(EXE_DIR)/%.cxx=$(PWD)/%.run)

$(EXE_OBJ): $(EXE_DIR)/%.o: $(EXE_DIR)/%.cxx
	$(CXX) -c -o $@ $< $(CPPFLAGS)

$(EXE_RUN): $(PWD)/%.run : $(EXE_DIR)/%.o
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(EXE_DIR)/*.o $(PWD)/*.run

all: $(EXE_RUN)

.DEFAULT_GOAL := all
