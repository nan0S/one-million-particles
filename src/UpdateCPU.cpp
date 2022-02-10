#include "UpdateCPU.h"

#include <thread>

#include "Utils/Error.h"
#include "Utils/Timer.h"
#include "Utils/Log.h"

namespace CPU
{
   /* macros */
   #define NTHREADS 8
   
   /* structs */
   struct SimState
   {
      size_t n_particles;
      size_t n_per_thread;
      SimMemory mem;
   };

   /* forward declarations */
   static void updateParticlesLocal(const size_t begin, const size_t end,
                                    const SimUpdate* sim_update);

   /* variables */
   static SimState sim_state;

   void init(const SimConfig* sim_config)
   {
      sim_state.n_particles = sim_config->n_particles;
      sim_state.n_per_thread = sim_state.n_particles / NTHREADS;
      /* Allocate buffers. */
      {
         GLbitfield flags = GL_MAP_WRITE_BIT | GL_MAP_READ_BIT |
                            GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
         size_t buffer_size = getTotalBufferSize(sim_state.n_particles);
         GL_CALL(glBufferStorage(GL_ARRAY_BUFFER, buffer_size, 0, flags));
         GL_CALL(void* buffer = glMapBufferRange(GL_ARRAY_BUFFER, 0, buffer_size, flags));
         sim_state.mem = getMemoryFromBuffer(buffer, sim_state.n_particles);
      }
      generateParticlesOnCPU(sim_config, &sim_state.mem);
   }

   void updateParticles(const SimUpdate* sim_update)
   {
#ifdef MEASURE_TIME
      static AggregateTimer timer("CPU::updateParticles");
      timer.start();
#endif
      /* Update particles. */
      {
         static std::thread threads[NTHREADS];
         size_t begin = 0, end = sim_state.n_per_thread;
         for (size_t i = 0; i < NTHREADS; ++i)
         {
            threads[i] = std::thread(updateParticlesLocal, begin, end, sim_update);
            begin = end;
            end += sim_state.n_per_thread;
         }
         updateParticlesLocal(begin, sim_state.n_particles, sim_update);
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
                             const SimUpdate* sim_update)
   {
      for (size_t i = begin; i < end; ++i)
      {
         vec2 p = sim_state.mem.pos[i];
         vec2 v = sim_state.mem.vel[i];
         float im = sim_state.mem.imass[i];

         vec2 f = sim_update->mouse_pos - p;
         float f_mult = im * sim_update->pull_strength * sim_update->delta_time;
         v += f_mult * f;
         if (sim_update->is_local_exp)
            v -= (im * sim_update->local_exp_strength / magnitude(f)) * f;
         if (sim_update->is_global_exp)
            v -= (im * sim_update->global_exp_strength / length(f)) * f;
         v *= sim_update->damp;
         
         float v_mult = sim_update->speed_mult * sim_update->delta_time;
         p += v_mult * v;

         sim_state.mem.pos[i] = p;
         sim_state.mem.vel[i] = v;
      }
   }

} // namespace CPU
