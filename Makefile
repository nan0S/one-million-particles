TARGET    := OneMillionParticles

SRC_DIR   := src
BUILD_DIR := build

INC       := -I$(CUDA_HOME)/include -I./$(SRC_DIR)
LIB       := -L$(CUDA_HOME)/lib64 -lcudart -lcurand -lGLEW -lGL -lglfw -lGLU -pthread

VERSION   := -std=c++17
CXXFLAGS  := $(VERSION) -Wall -pedantic
# CXXFLAGS  += -DNDEBUG -O3 -Wno-unused-variable -Wno-maybe-uninitialized #-DMEASURE_TIME
CXXFLAGS  += -ggdb -fno-omit-frame-pointer -O0 -DMEASURE_TIME
NVCCFLAGS := $(VERSION) -arch=sm_50 -Wno-deprecated-gpu-targets
# NVCCFLAGS += --use_fast_math --ptxas-options=-O3 #-DMEASURE_TIME
NVCCFLAGS += --ptxas-options=-O0 -DMEASURE_TIME

SRC := $(shell find $(SRC_DIR) -name '*.cpp')
OBJ := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRC))
DEP := $(patsubst %.cpp,$(BUILD_DIR)/%.d,$(SRC))
SRC := $(shell find $(SRC_DIR) -name '*.cu')
OBJ += $(patsubst %.cu,$(BUILD_DIR)/%.o,$(SRC))
DEP += $(patsubst %.cu,$(BUILD_DIR)/%.d,$(SRC))

.PHONY: clean

all: $(TARGET)

$(TARGET): $(OBJ)
	g++ $(CXXFLAGS) -o $@ $^ $(LIB)

-include $(DEP)

$(BUILD_DIR)/%.o: %.cpp Makefile
	@mkdir -p $(shell dirname $@)
	g++ -MMD -MP $(CXXFLAGS) -c -o $@ $< $(INC)

$(BUILD_DIR)/%.o: %.cu Makefile
	@mkdir -p $(shell dirname $@)
	nvcc -MMD -MP $(NVCCFLAGS) -c -o $@ $< $(INC)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) compile_commands.json

