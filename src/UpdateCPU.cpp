#include "UpdateCPU.h"

#include <random>
#include <thread>

#include "Utils/Error.h"
#include "Utils/Timer.h"
#include "Utils/Log.h"

namespace CPU
{
   /* macros */
   #define NTHREADS 8
   
   /* structs */
   struct SimulationState
   {
      vec2* pos;
      vec2* vel;
      float* imass;
   };

   /* forward declarations */
   static void updateParticlesLocal(const size_t begin, const size_t end,
                                    const vec2 mouse_pos, const float dt,
                                    const float pull_strength,
                                    const float speed_mult, const float damp,
                                    const bool is_local_exp,
                                    const bool is_global_exp,
                                    const float local_exp_strength,
                                    const float global_exp_strength);

   /* variables */
   static SimulationState state;

   void init(const size_t n_particles, const float mass_min,
             const float mass_max, const unsigned long long seed)
   {
      /* Allocate buffers. */
      {
         GLbitfield flags = GL_MAP_WRITE_BIT | GL_MAP_READ_BIT |
                            GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
         size_t total_memory = n_particles * (2 * sizeof(vec2) + sizeof(float));
         GL_CALL(glBufferStorage(GL_ARRAY_BUFFER, total_memory, 0, flags));
         GL_CALL(void* buffer = glMapBufferRange(GL_ARRAY_BUFFER, 0,
                                                 total_memory, flags));
         state.pos = reinterpret_cast<vec2*>(buffer);
         state.vel = reinterpret_cast<vec2*>(state.pos + n_particles);
         state.imass = reinterpret_cast<float*>(state.vel + n_particles);
      }
      /* Generate particles. */
      {
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
      }
   }

   void updateParticles(const size_t n_particles, const vec2 mouse_pos,
                        const float dt, const float pull_strength,
                        const float speed_mult, const float damp,
                        const bool is_local_exp, const bool is_global_exp,
                        const float local_exp_strength,
                        const float global_exp_strength)
   {
#ifdef MEASURE_TIME
      static AggregateTimer timer("CPU::updateParticles");
      timer.start();
#endif
      /* Update particles. */
      {
         static std::thread threads[NTHREADS];
         size_t n_per_thread = n_particles / NTHREADS;
         size_t begin = 0, end = n_per_thread;
         for (size_t i = 0; i < NTHREADS; ++i)
         {
            threads[i] = std::thread(updateParticlesLocal, begin, end, 
                                     mouse_pos, dt, pull_strength, speed_mult,
                                     damp, is_local_exp, is_global_exp,
                                     local_exp_strength, global_exp_strength);
            begin = end;
            end += n_per_thread;
         }
         updateParticlesLocal(begin, n_particles, mouse_pos, dt,
                              pull_strength, speed_mult, damp, is_local_exp,
                              is_global_exp, local_exp_strength, global_exp_strength);
         for (int i = 0; i < NTHREADS; ++i)
            threads[i].join();
      }

      /* Sync with OpenGL. */
      {
         GL_CALL(GLsync sync_obj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0));
         do {
            GL_CALL(GLenum res = glClientWaitSync(sync_obj, GL_SYNC_FLUSH_COMMANDS_BIT, 1));
            if (res == GL_ALREADY_SIGNALED || res == GL_CONDITION_SATISFIED)
               break;
         } while (true);
         GL_CALL(glDeleteSync(sync_obj));
      }
#ifdef MEASURE_TIME
      timer.stop();
#endif
   }

   void updateParticlesLocal(const size_t begin, const size_t end,
                             const vec2 mouse_pos, const float dt,
                             const float pull_strength, const float speed_mult,
                             const float damp, const bool is_local_exp,
                             const bool is_global_exp,
                             const float local_exp_strength,
                             const float global_exp_strength)
   {
      for (size_t i = begin; i < end; ++i)
      {
         vec2 p = state.pos[i];
         vec2 v = state.vel[i];
         float im = state.imass[i];

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

         state.pos[i] = p;
         state.vel[i] = v;
      }
   }

} // namespace CPU
