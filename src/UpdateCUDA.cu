#include "UpdateCUDA.h"

#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "Utils/Error.h"
#include "Utils/Timer.h"
#include "Utils/Log.h"

namespace CUDA
{
   /* macros */
   #define PI 3.14159265358f
   #define NTHREADS 1024
   #define NPERTHREAD_INIT 1024
   #define NPERTHREAD_UPDATE 12

   /* structs */
   struct Memory
   {
      vec2* pos;
      vec2* vel;
      float* imass;
   };

   struct SimulationState
   {
      cudaGraphicsResource_t resource;
      Memory mem;
   };
   
   /* forward declarations */
   inline size_t iceil(size_t x, size_t d) { return (x + d - 1) / d; }

   __global__
   void generateParticles(const Memory mem, const int N, const float imass_min,
                          const float imass_diff, const unsigned long long seed);
   __device__
   vec2 getRandomUnitVector(curandState* state);
   __global__
   void updateParticles(const Memory mem, const int N, const vec2 mouse_pos,
                        const float dt, const float pull_strength,
                        const float speed_mult, const float damp,
                        const bool is_local_exp, const bool is_global_exp,
                        const float local_exp_strength,
                        const float global_exp_strength);

   /* variables */
   static SimulationState state;

   void init(GLuint vbo, const size_t n_particles, const float mass_min,
             const float mass_max, const unsigned long long seed)
   {
      /* Allocate buffer. */
      {
         size_t buffer_size = n_particles * (2 * sizeof(vec2) + sizeof(float));
         GL_CALL(glBufferData(GL_ARRAY_BUFFER, buffer_size, NULL, GL_DYNAMIC_DRAW));
         CUDA_CALL(cudaGraphicsGLRegisterBuffer(&state.resource, vbo,
                                                cudaGraphicsRegisterFlagsNone));
      }
      /* Map buffers. */
      {
         CUDA_CALL(cudaGraphicsMapResources(1, &state.resource));
         void* d_buffer;
         size_t size;
         CUDA_CALL(cudaGraphicsResourceGetMappedPointer(&d_buffer, &size,
                                                        state.resource));
         state.mem.pos = reinterpret_cast<vec2*>(d_buffer);
         state.mem.vel = reinterpret_cast<vec2*>(state.mem.pos + n_particles);
         state.mem.imass = reinterpret_cast<float*>(state.mem.vel + n_particles);
      }
      /* Generate particles. */
      {
         size_t spawn_total = iceil(n_particles, NPERTHREAD_INIT);
         size_t nblocks = iceil(spawn_total, NTHREADS);
         float imass_min = 1.0f / mass_max, imass_max = 1.0f / mass_min;
         float imass_diff = imass_max - imass_min;
         CUDA_CALL(generateParticles<<<nblocks, NTHREADS>>>(state.mem, n_particles,
                                                            imass_min, imass_diff,
                                                            seed));
         CUDA_CALL(cudaDeviceSynchronize());
      }
      CUDA_CALL(cudaGraphicsUnmapResources(1, &state.resource));
   }

   void updateParticles(const size_t n_particles, const vec2 mouse_pos,
                        const float dt, const float pull_strength,
                        const float speed_mult, const float damp,
                        const bool is_local_exp, const bool is_global_exp,
                        const float local_exp_strength,
                        const float global_exp_strength)
   {
#ifdef MEASURE_TIME
      static AggregateTimer timer("CUDA::updateParticles");
      timer.start();
#endif
      CUDA_CALL(cudaGraphicsMapResources(1, &state.resource));
      size_t spawn_total = iceil(n_particles, NPERTHREAD_UPDATE);
      size_t nblocks = iceil(spawn_total, NTHREADS);
      CUDA_CALL(updateParticles<<<nblocks, NTHREADS>>>(state.mem,
                                                       n_particles, mouse_pos, dt,
                                                       pull_strength, speed_mult,
                                                       damp, is_local_exp, is_global_exp,
                                                       local_exp_strength,
                                                       global_exp_strength));
      CUDA_CALL(cudaDeviceSynchronize());
      CUDA_CALL(cudaGraphicsUnmapResources(1, &state.resource));
#ifdef MEASURE_TIME
      timer.stop();
#endif
   }
   
   void cleanup()
   {
      CUDA_CALL(cudaGraphicsUnregisterResource(state.resource));
      CUDA_CALL(cudaDeviceReset());
   }

   __global__
   void generateParticles(const Memory mem, const int N, const float imass_min,
                          const float imass_diff, const unsigned long long seed)
   {
      size_t idx = threadIdx.x + NPERTHREAD_INIT * blockIdx.x * blockDim.x;
      curandState state;
      curand_init(seed, idx, 0, &state);
      for (int i = 0; i < NPERTHREAD_INIT && idx < N; ++i)
      {
         mem.pos[idx] = getRandomUnitVector(&state);
         mem.vel[idx] = getRandomUnitVector(&state);
         mem.imass[idx] = imass_min + imass_diff * curand_uniform(&state);
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
   void updateParticles(const Memory mem, const int N, const vec2 mouse_pos,
                        const float dt, const float pull_strength,
                        const float speed_mult, const float damp,
                        const bool is_local_exp, const bool is_global_exp,
                        const float local_exp_strength,
                        const float global_exp_strength)
   {
      size_t idx = threadIdx.x + NPERTHREAD_UPDATE * blockIdx.x * blockDim.x;
      for (int i = 0; i < NPERTHREAD_UPDATE && idx < N; ++i)
      {
         vec2 p = mem.pos[idx];
         vec2 v = mem.vel[idx];
         float im = mem.imass[idx];

         /* Apply force to velocity. */
         {
            vec2 f = mouse_pos - p;
            float f_mult = im * pull_strength * dt;
            v += f_mult * f;
            if (is_local_exp)
               v -= (im * local_exp_strength / magnitude(f)) * f;
            if (is_global_exp)
               v -= (im * global_exp_strength / length(f)) * f;
            v *= damp;
         }
         /* Apply velocity to position. */
         {
            float v_mult = speed_mult * dt;
            p += v_mult * v;
         }

         mem.pos[idx] = p;
         mem.vel[idx] = v;

         idx += blockDim.x;
      }
   }

} // namespace CUDA
