#pragma once

#include <GL/glew.h>

#include "Math.cuh"

namespace CPU
{
   struct State {
      size_t total_memory;
      void* buffer;
      vec2* pos;
      vec2* vel;
      float* imass;
   };

   State init(GLuint vbo, const size_t n_particles, const float mass_min,
              const float mass_max, const unsigned long long seed);
   void updateParticles(State* state, GLuint vbo, const size_t n_particles,
                        const vec2 mouse_pos, const float dt,
                        const float pull_strength, const float speed_mult,
                        const float damp, const bool is_local_exp,
                        const bool is_global_exp,
                        const float local_exp_strength,
                        const float global_exp_strength);
   void cleanup(State* state);
}