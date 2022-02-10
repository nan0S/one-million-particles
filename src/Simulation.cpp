#include "Simulation.h"

#include <random>

size_t getTotalBufferSize(const size_t n_particles)
{
   return n_particles * (2 * sizeof(vec2) + sizeof(float));
}

SimMemory getMemoryFromBuffer(void* buffer, const size_t n_particles)
{
   SimMemory sim_mem;
   sim_mem.pos = reinterpret_cast<vec2*>(buffer);
   sim_mem.vel = reinterpret_cast<vec2*>(sim_mem.pos + n_particles);
   sim_mem.imass = reinterpret_cast<float*>(sim_mem.vel + n_particles);
   return sim_mem;
}

void generateParticlesOnCPU(const SimConfig* sim_config, const SimMemory* sim_mem)
{
   std::mt19937 rng(sim_config->seed);
   std::uniform_real_distribution<float> adist(0, 2 * PI);
   std::uniform_real_distribution<float> ldist(0, 1);
   float imass_min = 1.0f / sim_config->mass_max;
   float imass_max = 1.0f / sim_config->mass_min;
   std::uniform_real_distribution<float> imdist(imass_min, imass_max);
   auto getRandomUnitVector = [&rng, &adist, &ldist](){
      float angle = adist(rng);
      float len = ldist(rng);
      return len * vec2{ std::cos(angle), std::sin(angle) };
   };
   for (size_t i = 0; i < sim_config->n_particles; ++i)
   {
      sim_mem->pos[i] = getRandomUnitVector();
      sim_mem->vel[i] = getRandomUnitVector();
      sim_mem->imass[i] = imdist(rng);
   }
}
