#include "UpdateCPU.h"

#include <random>

#include "Utils/Error.h"

namespace CPU
{
   /* macros */
   #define PI 3.14159265358f

   State init(GLuint vbo, const size_t n_particles, const float mass_min,
              const float mass_max, const unsigned long long seed)
   {
      State state;
      state.total_memory = n_particles * (2 * sizeof(vec2) + sizeof(float));
      state.buffer = malloc(state.total_memory);
      state.pos = reinterpret_cast<vec2*>(state.buffer);
      state.vel = reinterpret_cast<vec2*>(state.pos + n_particles);
      state.imass = reinterpret_cast<float*>(state.vel + n_particles);
      std::mt19937 rng(seed);
      std::uniform_real_distribution<float> adist(0, 2 * PI);
      std::uniform_real_distribution<float> ldist(0, 1);
      float imass_min = 1.0f / mass_max, imass_max = 1.0f / mass_min;
      std::uniform_real_distribution<float> imdist(imass_min, imass_max);
      auto getRandomUnitVector = [&rng, &adist, &ldist](){
         float angle = adist(rng);
         float len = ldist(rng);
         return len * vec2{ std::cos(angle), std::sin(angle) };
      };
      for (size_t i = 0; i < n_particles; ++i)
      {
         state.pos[i] = getRandomUnitVector();
         state.vel[i] = getRandomUnitVector();
         state.imass[i] = imdist(rng);
      }
      GL_CALL(glBufferSubData(GL_ARRAY_BUFFER, 0, state.total_memory, state.buffer));
      return state;
   }

   void updateParticles(State* state, GLuint vbo, const size_t n_particles,
                        const vec2 mouse_pos, const float dt,
                        const float pull_strength, const float speed_mult,
                        const float damp, const bool is_local_exp,
                        const bool is_global_exp,
                        const float local_exp_strength,
                        const float global_exp_strength)
   {
      for (size_t i = 0; i < n_particles; ++i)
      {
         vec2 p = state->pos[i];
         vec2 v = state->vel[i];
         float im = state->imass[i];

         vec2 f = mouse_pos - p;
         float f_mult = im * pull_strength * dt;
         v += f_mult * f;
         if (is_local_exp)
            v -= (im * local_exp_strength / magnitude(f)) * f;
         if (is_global_exp)
            v -= (im * global_exp_strength / length(f)) * f;
         v *= damp;
         
         float v_mult = speed_mult * dt;
         p += v_mult * v;

         state->pos[i] = p;
         state->vel[i] = v;
      }
      GL_CALL(glBufferSubData(GL_ARRAY_BUFFER, 0, state->total_memory, state->buffer));
   }

   void cleanup(State* state)
   {
      free(state->buffer);
   }

} // namespace CPU
