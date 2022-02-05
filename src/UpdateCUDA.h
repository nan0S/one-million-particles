#pragma once

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "Math.cuh"

namespace CUDA
{
   struct State
   {
      cudaGraphicsResource_t resource;
   };

   State init(GLuint vbo, const size_t n_particles, const float mass_min,
              const float mass_max, const unsigned long long seed);
   void updateParticles(State* state, const size_t n_particles,
                        const vec2 mouse_pos, const float dt,
                        const float pull_strength, const float speed_mult,
                        const float damp, const bool is_local_exp,
                        const bool is_global_exp,
                        const float local_exp_strength,
                        const float global_exp_strength);
   void cleanup(State* state);
}
