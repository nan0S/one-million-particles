#include "UpdateCUDA.h"

#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "Utils/Error.h"
#include "Utils/Timer.h"
#include "Utils/Log.h"

namespace CUDA
{
   /* macros */
   #define NTHREADS 1024
   #define NPERTHREAD_INIT 1024
   #define NPERTHREAD_UPDATE 12

   /* structs */
   struct SimState
   {
      size_t n_particles;
      cudaGraphicsResource_t resource;
      SimMemory mem;
   };
   
   /* forward declarations */
   __global__
   void generateParticles(const SimConfig sim_config, const SimMemory mem);
   __device__
   vec2 getRandomUnitVector(curandState* state);
   __global__
   void updateParticles(const size_t n_particles, const SimMemory mem,
                        const SimUpdate sim_update);

   /* variables */
   static SimState sim_state;

   void init(const SimConfig* sim_config, GLuint vbo)
   {
      sim_state.n_particles = sim_config->n_particles;
      /* Allocate buffer. */
      {
         size_t buffer_size = sim_config->n_particles * (2 * sizeof(vec2) +
                                                         sizeof(float));
         GL_CALL(glBufferData(GL_ARRAY_BUFFER, buffer_size, NULL, GL_DYNAMIC_DRAW));
         CUDA_CALL(cudaGraphicsGLRegisterBuffer(&sim_state.resource, vbo,
                                                cudaGraphicsRegisterFlagsNone));
      }
      /* Map buffers. */
      {
         CUDA_CALL(cudaGraphicsMapResources(1, &sim_state.resource));
         void* d_buffer;
         size_t size;
         CUDA_CALL(cudaGraphicsResourceGetMappedPointer(&d_buffer, &size,
                                                        sim_state.resource));
         sim_state.mem = getMemoryFromBuffer(d_buffer, sim_state.n_particles);
      }
      /* Generate particles. */
      {
         size_t spawn_total = iceil(sim_config->n_particles, NPERTHREAD_INIT);
         size_t nblocks = iceil(spawn_total, NTHREADS);
         float imass_min = 1.0f / sim_config->mass_max;
         float imass_max = 1.0f / sim_config->mass_min;
         float imass_diff = imass_max - imass_min;
         SimConfig cuda_sim_config = *sim_config;
         cuda_sim_config.mass_min = imass_min;
         cuda_sim_config.mass_max = imass_diff;
         CUDA_CALL(generateParticles<<<nblocks, NTHREADS>>>(cuda_sim_config,
                                                            sim_state.mem));
      }
      CUDA_CALL(cudaGraphicsUnmapResources(1, &sim_state.resource));
   }

   void updateParticles(const SimUpdate* sim_update)
   {
#ifdef MEASURE_TIME
      static AggregateTimer timer("CUDA::updateParticles");
      timer.start();
#endif
      CUDA_CALL(cudaGraphicsMapResources(1, &sim_state.resource));
      size_t spawn_total = iceil(sim_state.n_particles, NPERTHREAD_UPDATE);
      size_t nblocks = iceil(spawn_total, NTHREADS);
      CUDA_CALL(updateParticles<<<nblocks, NTHREADS>>>(sim_state.n_particles,
                                                       sim_state.mem, *sim_update));
      CUDA_CALL(cudaGraphicsUnmapResources(1, &sim_state.resource));
#ifdef MEASURE_TIME
      timer.stop();
#endif
   }
   
   void cleanup()
   {
      CUDA_CALL(cudaGraphicsUnregisterResource(sim_state.resource));
      CUDA_CALL(cudaDeviceReset());
   }

   __global__
   void generateParticles(const SimConfig sim_config, const SimMemory mem)
   {
      size_t idx = threadIdx.x + NPERTHREAD_INIT * blockIdx.x * blockDim.x;
      curandState state;
      curand_init(sim_config.seed, idx, 0, &state);
      for (int i = 0; i < NPERTHREAD_INIT && idx < sim_config.n_particles; ++i)
      {
         mem.pos[idx] = getRandomUnitVector(&state);
         mem.vel[idx] = getRandomUnitVector(&state);
         mem.imass[idx] = sim_config.mass_min + sim_config.mass_max *
            curand_uniform(&state);
         idx += blockDim.x;
      }
   }

   __device__
   vec2 getRandomUnitVector(curandState* state)
   {
      float angle = 2 * PI * curand_uniform(state);
      float len = curand_uniform(state);
      return len * vec2{ cos(angle), sin(angle) };
   }

   __global__
   void updateParticles(const size_t n_particles, const SimMemory mem,
                        const SimUpdate sim_update)
   {
      size_t idx = threadIdx.x + NPERTHREAD_UPDATE * blockIdx.x * blockDim.x;
      for (int i = 0; i < NPERTHREAD_UPDATE && idx < n_particles; ++i)
      {
         vec2 p = mem.pos[idx];
         vec2 v = mem.vel[idx];
         float im = mem.imass[idx];

         /* Apply force to velocity. */
         {
            vec2 f = sim_update.mouse_pos - p;
            float f_mult = im * sim_update.pull_strength * sim_update.delta_time;
            v += f_mult * f;
            if (sim_update.is_local_exp)
               v -= (im * sim_update.local_exp_strength / magnitude(f)) * f;
            if (sim_update.is_global_exp)
               v -= (im * sim_update.global_exp_strength / length(f)) * f;
            v *= sim_update.damp;
         }
         /* Apply velocity to position. */
         {
            float v_mult = sim_update.speed_mult * sim_update.delta_time;
            p += v_mult * v;
         }

         mem.pos[idx] = p;
         mem.vel[idx] = v;

         idx += blockDim.x;
      }
   }

} // namespace CUDA
