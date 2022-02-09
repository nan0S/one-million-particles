#include "UpdateCompute.h"

// TODO: remove
#include <random>

#include "Graphics/Shader.h"
#include "Utils/Error.h"

namespace Compute
{
   /* structs */
   struct SimulationState
   {
      GLuint shader;
      size_t n_groups;
      GLint n_particles_loc;
      GLint mouse_pos_loc;
      GLint dt_loc;
      GLint pull_strength_loc;
      GLint speed_mult_loc;
      GLint damp_loc;
      GLint is_local_exp_loc;
      GLint is_global_exp_loc;
      GLint local_exp_strength_loc;
      GLint global_exp_strength_loc;
   };

   /* constants */
   static const char* COMPUTE_SHADER_SOURCE =
   R"(
   #version 430

   layout (local_size_x = 128) in;

   layout (std430, binding = 0) buffer Pos
   {
      vec2 pos[];
   };
   layout (std430, binding = 1) buffer Vel
   {
      vec2 vel[];
   };
   layout (std430, binding = 2) buffer IMass
   {
      float imass[];
   };

   uniform int n_particles;
   uniform vec2 mouse_pos;
   uniform float dt;
   uniform float pull_strength;
   uniform float speed_mult;
   uniform float damp;
   uniform bool is_local_exp;
   uniform bool is_global_exp;
   uniform float local_exp_strength;
   uniform float global_exp_strength;

   void main()
   {
      uint idx = gl_GlobalInvocationID.x;
      if (idx < n_particles)
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
               v -= (im * local_exp_strength / length(f) / length(f)) * f;
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
      }
   }
   )";

   static constexpr size_t GROUP_SIZE = 128;

   /* variables */
   static SimulationState state;

   void init(GLuint vbo, const size_t n_particles, const float mass_min,
             const float mass_max, const unsigned long long seed)
   {
      /* Setup state. */
      state.shader = Graphics::createComputeShader(COMPUTE_SHADER_SOURCE);
      state.n_groups = (n_particles + GROUP_SIZE - 1) / GROUP_SIZE;
      GL_CALL(state.n_particles_loc = glGetUniformLocation(state.shader, "n_particles"));
      GL_CALL(state.mouse_pos_loc = glGetUniformLocation(state.shader, "mouse_pos"));
      GL_CALL(state.dt_loc = glGetUniformLocation(state.shader, "dt"));
      GL_CALL(state.pull_strength_loc = glGetUniformLocation(state.shader,"pull_strength"));
      GL_CALL(state.speed_mult_loc = glGetUniformLocation(state.shader, "speed_mult"));
      GL_CALL(state.damp_loc = glGetUniformLocation(state.shader, "damp"));
      GL_CALL(state.is_local_exp_loc = glGetUniformLocation(state.shader, "is_local_exp"));
      GL_CALL(state.is_global_exp_loc = glGetUniformLocation(state.shader, "is_global_exp"));
      GL_CALL(state.local_exp_strength_loc = glGetUniformLocation(state.shader, "local_exp_strength"));
      GL_CALL(state.global_exp_strength_loc = glGetUniformLocation(state.shader, "global_exp_strength"));
      /* Generate points. */
      {
         size_t buffer_size = n_particles * (2 * sizeof(vec2) + sizeof(float));
         void* buffer = malloc(buffer_size);
         // GL_CALL(glBufferData(GL_ARRAY_BUFFER, buffer_size, NULL, GL_DYNAMIC_DRAW));
         // GL_CALL(void* buffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
         vec2* pos = (vec2*)buffer;
         vec2* vel = pos + n_particles;
         float* imass = (float*)(vel + n_particles);
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
            pos[i] = getRandomUnitVector();
            vel[i] = getRandomUnitVector();
            imass[i] = imdist(rng);
         }
         GL_CALL(glBufferData(GL_ARRAY_BUFFER, buffer_size, buffer, GL_DYNAMIC_DRAW));
         free(buffer);
         // GL_CALL(glUnmapBuffer(GL_ARRAY_BUFFER));
         // TODO: commented code (instread of malloc ...) makes the WHOLE program
         //       to run slower (WTF)
      }
   }

   void updateParticles(GLuint render_shader, GLuint vbo,
                        const size_t n_particles, const vec2 mouse_pos,
                        const float dt, const float pull_strength,
                        const float speed_mult, const float damp,
                        const bool is_local_exp, const bool is_global_exp,
                        const float local_exp_strength,
                        const float global_exp_strength)
   {
      /* Set uniforms. */
      GL_CALL(glUseProgram(state.shader));
      GL_CALL(glUniform1i(state.n_particles_loc, n_particles));
      GL_CALL(glUniform2fv(state.mouse_pos_loc, 1, (GLfloat*)&mouse_pos));
      GL_CALL(glUniform1f(state.dt_loc, dt));
      GL_CALL(glUniform1f(state.pull_strength_loc, pull_strength));
      GL_CALL(glUniform1f(state.speed_mult_loc, speed_mult));
      GL_CALL(glUniform1f(state.damp_loc, damp));
      GL_CALL(glUniform1i(state.is_local_exp_loc, is_local_exp));
      GL_CALL(glUniform1i(state.is_global_exp_loc, is_global_exp));
      GL_CALL(glUniform1f(state.local_exp_strength_loc, local_exp_strength));
      GL_CALL(glUniform1f(state.global_exp_strength_loc, global_exp_strength));
      /* Bind data and dispatch computation. */
      GL_CALL(glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, vbo,
                                0, n_particles * sizeof(vec2)));
      GL_CALL(glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, vbo,
                                n_particles * sizeof(vec2),
                                n_particles * sizeof(vec2)));
      GL_CALL(glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, vbo,
                                2 * n_particles * sizeof(vec2),
                                n_particles * sizeof(float)));
      GL_CALL(glDispatchCompute(state.n_groups, 1, 1));
      GL_CALL(glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT));
      GL_CALL(glUseProgram(render_shader));
   }

   void cleanup()
   {
      GL_CALL(glDeleteProgram(state.shader));
   }

} // namespace Compute
