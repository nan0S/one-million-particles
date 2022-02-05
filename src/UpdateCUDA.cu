#include "UpdateCUDA.h"

#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "Utils/Error.h"

namespace CUDA
{
   /* macros */
   #define PI 3.14159265358f
   #define NTHREADS 1024
   #define NPERTHREAD_INIT 1024
   #define NPERTHREAD_UPDATE 12
   
   /* forward declarations */
   inline size_t iceil(size_t x, size_t d) { return (x + d - 1) / d; }

   __global__
   void generateParticles(vec2* pos, vec2* vel, float* imass, const int N,
                          const float imass_min, const float imass_diff,
                          const unsigned long long seed);
   __device__
   vec2 getRandomUnitVector(curandState* state);
   __global__
   void updateParticles(vec2* pos, vec2* vel, float* imass, const int N,
                        vec2 mouse_pos, const float dt, const float pull_strength,
                        const float speed_mult, const float damp,
                        const bool is_local_exp, const bool is_global_exp,
                        const float local_exp_strength,
                        const float global_exp_strength);

   State init(GLuint vbo, const size_t n_particles, const float mass_min,
              const float mass_max, const unsigned long long seed)
   {
      State state;
      CUDA_CALL(cudaGraphicsGLRegisterBuffer(&state.resource, vbo,
                                             cudaGraphicsRegisterFlagsNone));
      CUDA_CALL(cudaGraphicsMapResources(1, &state.resource));
      void* d_buffer;
      size_t size;
      CUDA_CALL(cudaGraphicsResourceGetMappedPointer(&d_buffer, &size, state.resource));
      vec2* pos = reinterpret_cast<vec2*>(d_buffer);
      vec2* vel = reinterpret_cast<vec2*>(pos + n_particles);
      float* imass = reinterpret_cast<float*>(vel + n_particles);
      size_t spawn_total = iceil(n_particles, NPERTHREAD_INIT);
      size_t nblocks = iceil(spawn_total, NTHREADS);
      float imass_min = 1.0f / mass_max, imass_max = 1.0f / mass_min;
      float imass_diff = imass_max - imass_min;
      CUDA_CALL(generateParticles<<<nblocks, NTHREADS>>>(pos, vel, imass, n_particles,
                                                         imass_min, imass_diff, seed));
      CUDA_CALL(cudaDeviceSynchronize());
      CUDA_CALL(cudaGraphicsUnmapResources(1, &state.resource));
      return state;
   }

   void updateParticles(State* state, const size_t n_particles,
                        const vec2 mouse_pos, const float dt,
                        const float pull_strength, const float speed_mult,
                        const float damp, const bool is_local_exp,
                        const bool is_global_exp,
                        const float local_exp_strength,
                        const float global_exp_strength)
   {
      CUDA_CALL(cudaGraphicsMapResources(1, &state->resource));
      void* d_buffer;
      size_t size;
      CUDA_CALL(cudaGraphicsResourceGetMappedPointer(&d_buffer, &size, state->resource));
      vec2* pos = reinterpret_cast<vec2*>(d_buffer);
      vec2* vel = reinterpret_cast<vec2*>(pos + n_particles);
      float* imass = reinterpret_cast<float*>(vel + n_particles);
      size_t spawn_total = iceil(n_particles, NPERTHREAD_UPDATE);
      size_t nblocks = iceil(spawn_total, NTHREADS);
      CUDA_CALL(updateParticles<<<nblocks, NTHREADS>>>(pos, vel, imass,
                                                       n_particles, mouse_pos, dt,
                                                       pull_strength, speed_mult,
                                                       damp, is_local_exp, is_global_exp,
                                                       local_exp_strength,
                                                       global_exp_strength));
      CUDA_CALL(cudaDeviceSynchronize());
      CUDA_CALL(cudaGraphicsUnmapResources(1, &state->resource));
   }
   
   void cleanup(State* state)
   {
      CUDA_CALL(cudaGraphicsUnregisterResource(state->resource));
      CUDA_CALL(cudaDeviceReset());
   }

   __global__
   void generateParticles(vec2* pos, vec2* vel, float* imass, const int N,
                          const float imass_min, const float imass_diff,
                          const unsigned long long seed)
   {
      size_t idx = threadIdx.x + NPERTHREAD_INIT * blockIdx.x * blockDim.x;
      curandState state;
      curand_init(seed, idx, 0, &state);
      for (int i = 0; i < NPERTHREAD_INIT && idx < N; ++i)
      {
         pos[idx] = getRandomUnitVector(&state);
         vel[idx] = getRandomUnitVector(&state);
         imass[idx] = imass_min + imass_diff * curand_uniform(&state);
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
   void updateParticles(vec2* pos, vec2* vel, float* imass, const int N,
                        vec2 mouse_pos, const float dt, const float pull_strength,
                        const float speed_mult, const float damp,
                        const bool is_local_exp, const bool is_global_exp,
                        const float local_exp_strength,
                        const float global_exp_strength)
   {
      size_t idx = threadIdx.x + NPERTHREAD_UPDATE * blockIdx.x * blockDim.x;
      for (int i = 0; i < NPERTHREAD_UPDATE && idx < N; ++i)
      {
         vec2 p = pos[idx];
         vec2 v = vel[idx];
         float im = imass[idx];

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

         pos[idx] = p;
         vel[idx] = v;

         idx += blockDim.x;
      }
   }

} // namespace CUDA
