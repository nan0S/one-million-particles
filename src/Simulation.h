#pragma once

#include "Utils/Math.cuh"

struct SimConfig
{
   size_t n_particles;
   unsigned long long seed;
   float mass_min;
   float mass_max;
};

struct SimUpdate
{
   vec2 mouse_pos;
   float delta_time;
   float pull_strength;
   float speed_mult;
   float damp;
   bool is_local_exp;
   bool is_global_exp;
   float local_exp_strength;
   float global_exp_strength;
};

struct SimMemory
{
   vec2* pos;
   vec2* vel;
   float* imass;
};

size_t getTotalBufferSize(const size_t n_particles);
SimMemory getMemoryFromBuffer(void* buffer, const size_t n_particles);
void generateParticlesOnCPU(const SimConfig* sim_config, const SimMemory* sim_mem);

