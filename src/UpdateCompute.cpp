#include "UpdateCompute.h"
#include <iomanip>
#include <iostream>

#include "Graphics/Shader.h"
#include "Utils/Error.h"
#include "Utils/Timer.h"

namespace Compute
{
   /* macros */
   #define GROUP_SIZE 256

   /* structs */
   struct SimState
   {
      size_t n_particles;
      GLuint shader;
      size_t n_groups;
      GLint mouse_pos_loc;
      GLint delta_time_loc;
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

   layout (local_size_x = 256) in;

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
   uniform float delta_time;
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
            float f_mult = im * pull_strength * delta_time;
            v += f_mult * f;
            if (is_local_exp)
               v -= (im * local_exp_strength / length(f) / length(f)) * f;
            if (is_global_exp)
               v -= (im * global_exp_strength / length(f)) * f;
            v *= damp;
         }
         /* Apply velocity to position. */
         {
            float v_mult = speed_mult * delta_time;
            p += v_mult * v;
         }

         pos[idx] = p;
         vel[idx] = v;
      }
   }
   )";

   /* variables */
   static SimState sim_state;

   void init(const SimConfig* sim_config, GLuint vbo, GLuint render_shader)
   {
      /* Setup state. */
      {
         sim_state.n_particles = sim_config->n_particles;
         GLuint shader = Graphics::createComputeShader(COMPUTE_SHADER_SOURCE);
         sim_state.shader = shader;
         sim_state.n_groups = iceil(sim_state.n_particles, GROUP_SIZE);

         GL_CALL(glUseProgram(shader));
         GL_CALL(GLint n_particles_loc = glGetUniformLocation(shader, "n_particles"));
         GL_CALL(glUniform1i(n_particles_loc, sim_state.n_particles));
         GL_CALL(glUseProgram(render_shader));

         GL_CALL(sim_state.mouse_pos_loc = glGetUniformLocation(shader, "mouse_pos"));
         GL_CALL(sim_state.delta_time_loc = glGetUniformLocation(shader, "delta_time"));
         GL_CALL(sim_state.pull_strength_loc = glGetUniformLocation(shader,"pull_strength"));
         GL_CALL(sim_state.speed_mult_loc = glGetUniformLocation(shader, "speed_mult"));
         GL_CALL(sim_state.damp_loc = glGetUniformLocation(shader, "damp"));
         GL_CALL(sim_state.is_local_exp_loc = glGetUniformLocation(shader, "is_local_exp"));
         GL_CALL(sim_state.is_global_exp_loc = glGetUniformLocation(shader, "is_global_exp"));
         GL_CALL(sim_state.local_exp_strength_loc = glGetUniformLocation(shader, "local_exp_strength"));
         GL_CALL(sim_state.global_exp_strength_loc = glGetUniformLocation(shader, "global_exp_strength"));
      }
      /* Generate points. */
      {
         size_t buffer_size = getTotalBufferSize(sim_state.n_particles);
         // void* buffer = malloc(buffer_size);
         GL_CALL(glBufferData(GL_ARRAY_BUFFER, buffer_size, NULL, GL_DYNAMIC_COPY));
         GLbitfield flags = GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT;
         GL_CALL(void* buffer = glMapBufferRange(GL_ARRAY_BUFFER, 0, buffer_size, flags));
         SimMemory sim_mem = getMemoryFromBuffer(buffer, sim_state.n_particles);
         generateParticlesOnCPU(sim_config, &sim_mem);
         // GL_CALL(glBufferData(GL_ARRAY_BUFFER, buffer_size, buffer, GL_DYNAMIC_COPY));
         // free(buffer);
         GL_CALL(glUnmapBuffer(GL_ARRAY_BUFFER));
         // TODO: changing GL_DYNAMIC_COPY to GL_DYNAMIC_DRAW has major
         // performance implications for no apparent reason (on the other hand
         // using malloc and not glMap could be used with GL_DYNAMIC_DRAW without
         // performance costs)
      }
   }

   void updateParticles(const SimUpdate* sim_update, GLuint vbo,
                        GLuint render_shader)
   {
      /* Set uniforms. */
      GL_CALL(glUseProgram(sim_state.shader));
      GL_CALL(glUniform2fv(sim_state.mouse_pos_loc, 1, (GLfloat*)&sim_update->mouse_pos));
      GL_CALL(glUniform1f(sim_state.delta_time_loc, sim_update->delta_time));
      GL_CALL(glUniform1f(sim_state.pull_strength_loc, sim_update->pull_strength));
      GL_CALL(glUniform1f(sim_state.speed_mult_loc, sim_update->speed_mult));
      GL_CALL(glUniform1f(sim_state.damp_loc, sim_update->damp));
      GL_CALL(glUniform1i(sim_state.is_local_exp_loc, sim_update->is_local_exp));
      if (sim_update->is_local_exp)
         GL_CALL(glUniform1f(sim_state.local_exp_strength_loc, sim_update->local_exp_strength));
      GL_CALL(glUniform1i(sim_state.is_global_exp_loc, sim_update->is_global_exp));
      if (sim_update->is_global_exp)
         GL_CALL(glUniform1f(sim_state.global_exp_strength_loc, sim_update->global_exp_strength));
      /* Bind data and dispatch computation. */
      GL_CALL(glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, vbo,
                                0, sim_state.n_particles * sizeof(vec2)));
      GL_CALL(glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, vbo,
                                sim_state.n_particles * sizeof(vec2),
                                sim_state.n_particles * sizeof(vec2)));
      GL_CALL(glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, vbo,
                                2 * sim_state.n_particles * sizeof(vec2),
                                sim_state.n_particles * sizeof(float)));
      GL_CALL(glDispatchCompute(sim_state.n_groups, 1, 1));
      GL_CALL(glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT));
      GL_CALL(glUseProgram(render_shader));
   }

   void cleanup()
   {
      GL_CALL(glDeleteProgram(sim_state.shader));
   }

} // namespace Compute
