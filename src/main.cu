#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>
// TODO: remove
#include <random>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Utils/Log.h"
#include "Utils/Error.h"
#include "Graphics/Shader.h"
#include "Math.cuh"

/* macros */
#define PI 3.14159265358f
#define NTHREADS 1024
#define NPERTHREAD 1024

/* structs */
struct comma_numpunct : public std::numpunct<char>
{
protected:
   char do_thousands_sep() const { return ','; }
   std::string do_grouping() const { return "\03"; }
};

/* forward declarations */
void glfwErrorCallback(int code, const char* desc);
void windowResizeHandler(GLFWwindow*, int width, int height);
void windowKeyInputHandler(GLFWwindow* window, int key, int, int action, int);

__global__
void generateParticles(vec2* pos, vec2* vel, float* imass, const int N,
                       const float imass_min, const float imass_diff,
                       const unsigned long long seed);
__global__
void updateParticles(vec2* pos, vec2* vel, float* imass, const int N,
                     vec2 mpos, const float dt, const float pull_strength,
                     const float particle_speed, const float damp,
                     const bool is_clicked1, const bool is_clicked2,
                     const float click1_strength, const float click2_strength);

/* constants */
const char* USAGE_STR =
"Usage: ./OneMillionParticles [OPTION]... [N_PARTICLES]\n\n"
"Run n-particles simulation that followe mouse pointer.\n\n"
"list of possible options:\n"
"   --help        print this help";

const char* VERTEX_SHADER_SOURCE =
R"(
#version 330 core

layout (location = 0) in vec2 v_Pos;
layout (location = 1) in vec2 v_Vel;
layout (location = 2) in float v_Imass;

out vec4 v_Color;

void main()
{
   v_Color = vec4(1, 0, 0, 1);
   gl_Position = vec4(v_Pos, 0, 1);
   // gl_PointSize = 1 / v_Imass / 5;
}
)";

const char* FRAGMENT_SHADER_SOURCE =
R"(
#version 330 core

in vec4 v_Color;
out vec4 f_Color;

void main()
{
   f_Color = v_Color;
}
)";

constexpr int WINDOW_WIDTH = 1280;
constexpr int WINDOW_HEIGHT = 720;

constexpr float TIME_BETWEEN_FPS_REPORT = 0.2f;

constexpr size_t N_PARTICLES = 1'000'000;
constexpr float PULL_STRENGTH = 1.f;
constexpr float PARTICLE_SPEED = 1.f;
constexpr unsigned long long SEED = 1234ull;
constexpr float MASS_MIN = 0.5f;
constexpr float MASS_MAX = 10.0f;
constexpr float DAMP_FACTOR = 0.995f;
constexpr float DAMP_INTERVAL = 1.0f / 30;
constexpr float CLICK1_STRENGTH = 1.5f;
constexpr float CLICK2_STRENGTH = 3.0f;

/* variables */
int width, height;

int main(int argc, char* argv[])
{
   {
      std::locale comma_locale(std::locale(), new comma_numpunct);
      std::cout.imbue(comma_locale);
      std::cerr.imbue(comma_locale);
      std::cout << std::setprecision(2) << std::fixed;
   }

   assert(argc > 0);

   size_t n_particles = N_PARTICLES;
   float pull_strength = PULL_STRENGTH;
   float particle_speed = PARTICLE_SPEED;

   /* Parse program arguments */
   int pivot_idx = 1;
   for (int i = 1; i < argc; ++i)
   {
      const char* arg = argv[i];
      if (arg[0] != '-')
      {
         std::swap(argv[i], argv[pivot_idx++]);
         continue;
      }
      const char* flag = (arg[1] == '-' ? arg + 2 : arg + 1);
      if (strcmp(flag, "n_particles") == 0)
      {
         if (i == argc - 1)
            ERROR("Argument for flag '", arg, "' is required.");
         n_particles = std::stoull(argv[++i]);
      }
      else if (strcmp(flag, "pull_strength") == 0)
      {
         if (i == argc - 1)
            ERROR("Argument for flag '", arg, "' is required.");
         pull_strength = std::stof(argv[++i]);
      }
      else if (strcmp(flag, "particle_speed") == 0)
      {
         if (i == argc - 1)
            ERROR("Argument for flag '", arg, "' is required.");
         particle_speed = std::stof(argv[++i]);
      }
      else if (strcmp(flag, "help") == 0)
      {
         print(USAGE_STR);
         exit(EXIT_SUCCESS);
      }
      else
         ERROR("Flag '", arg, "' is no recognized.");
   }

   /* Setup GLFW and GLEW. */
   glfwSetErrorCallback(glfwErrorCallback);
   if (!glfwInit())
      ERROR("Failed to initialize GLFW.");

   glfwWindowHint(GLFW_SAMPLES, 4);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
   glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
   glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

   GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
                                         "OneMillionParticles", NULL, NULL);
   if (window == NULL)
   {
      glfwTerminate();
      ERROR("Failed to create GLFW window.");
   }

   glfwMakeContextCurrent(window);
   windowResizeHandler(window, WINDOW_WIDTH, WINDOW_HEIGHT);
   glfwSetFramebufferSizeCallback(window, windowResizeHandler);
   glfwSetKeyCallback(window, windowKeyInputHandler);
   glfwSwapInterval(0);

   if (glewInit() != GLEW_OK)
   {
      glfwTerminate();
      ERROR("Failed to initialize GLEW.");
   }

   GL_CALL(glClearColor(0, 0, 0, 1));
   GL_CALL(glClear(GL_COLOR_BUFFER_BIT));
   GL_CALL(glEnable(GL_VERTEX_PROGRAM_POINT_SIZE));

   GLuint vao;
   GL_CALL(glGenVertexArrays(1, &vao));
   GL_CALL(glBindVertexArray(vao));
   
   size_t total_memory = n_particles * (2 * sizeof(vec2) + sizeof(float));
   GLuint vbo;
   GL_CALL(glGenBuffers(1, &vbo));
   GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
   GL_CALL(glBufferData(GL_ARRAY_BUFFER,
                       total_memory,
                       NULL, GL_DYNAMIC_DRAW));
   GL_CALL(glEnableVertexAttribArray(0));
   GL_CALL(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                                sizeof(vec2), 0));
   GL_CALL(glEnableVertexAttribArray(1));
   GL_CALL(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                                sizeof(vec2),
                                reinterpret_cast<const void*>(n_particles * sizeof(vec2))));
   GL_CALL(glEnableVertexAttribArray(2));
   GL_CALL(glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE,
                                sizeof(float),
                                reinterpret_cast<const void*>(2 * n_particles * sizeof(vec2))));

   GLuint shader = Graphics::compileShader(VERTEX_SHADER_SOURCE,
                                           FRAGMENT_SHADER_SOURCE);
   GL_CALL(glUseProgram(shader));

   /* Setup CUDA. */
   cudaGraphicsResource_t resource;
   CUDA_CALL(cudaGraphicsGLRegisterBuffer(&resource, vbo,
                                         cudaGraphicsRegisterFlagsNone));

   /* Initialize particles. */
   auto iceil = [](size_t x, size_t d) { return (x + d - 1) / d; };
   {
      CUDA_CALL(cudaGraphicsMapResources(1, &resource));

      void* d_buffer;
      size_t size;
      CUDA_CALL(cudaGraphicsResourceGetMappedPointer(&d_buffer, &size, resource));
      assert(size == total_memory);
      vec2* pos = reinterpret_cast<vec2*>(d_buffer);
      vec2* vel = reinterpret_cast<vec2*>(pos + n_particles);
      float* imass = reinterpret_cast<float*>(vel + n_particles);

      size_t spawn_total = iceil(n_particles, NPERTHREAD);
      size_t nblocks = iceil(spawn_total, NTHREADS);
      float imass_min = 1.0f / MASS_MAX, imass_max = 1.0f / MASS_MIN;
      float imass_diff = imass_max - imass_min;
      CUDA_CALL(generateParticles<<<nblocks, NTHREADS>>>(pos, vel, imass, n_particles,
                                                         imass_min, imass_diff, SEED));
      CUDA_CALL(cudaDeviceSynchronize());

      CUDA_CALL(cudaGraphicsUnmapResources(1, &resource));
   }

   print("Running simulation for ", n_particles, " particles.");

   int frames_drawn = 0;
   float last_time = glfwGetTime();
   float last_fps_time = last_time;
   float last_damp_time = last_time;
   int last_click1_state = GLFW_RELEASE;
   int last_click2_state = GLFW_RELEASE;
   
   while (!glfwWindowShouldClose(window))
   {
      GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

      /* Handle input. */
      bool is_clicked1, is_clicked2;
      {
         int click1_state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1);
         is_clicked1 = click1_state == GLFW_PRESS && last_click1_state == GLFW_RELEASE;
         last_click1_state = click1_state;
         int click2_state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2);
         is_clicked2 = click2_state == GLFW_PRESS && last_click2_state == GLFW_RELEASE;
         last_click2_state = click2_state;
      }

      /* Get current mouse position. */
      vec2 mpos;
      {
         double x, y;
         glfwGetCursorPos(window, &x, &y);
         mpos.x = 2 * (x - 0.5f * width) / width;
         mpos.y = -2 * (y - 0.5f * height) / height;
      }

      /* Calculate delta time. */
      float dt;
      {
         float now = glfwGetTime();
         dt = now - last_time;
         last_time = now;
      }

      /* Calculate damp. */
      float damp;
      {
         float now = glfwGetTime();
         float passed = now - last_damp_time;
         if (passed >= DAMP_INTERVAL)
         {
            damp = DAMP_FACTOR;
            last_damp_time = now;
         }
         else
            damp = 1.0f;
      }

      /* Update particles. */
      {
         CUDA_CALL(cudaGraphicsMapResources(1, &resource));
         void* d_buffer;
         size_t size;
         CUDA_CALL(cudaGraphicsResourceGetMappedPointer(&d_buffer, &size, resource));
         assert(size == total_memory);
         vec2* pos = reinterpret_cast<vec2*>(d_buffer);
         vec2* vel = reinterpret_cast<vec2*>(pos + n_particles);
         float* imass = reinterpret_cast<float*>(vel + n_particles);
         size_t nblocks = iceil(n_particles, NTHREADS);
         CUDA_CALL(updateParticles<<<nblocks, NTHREADS>>>(pos, vel, imass,
                                                          n_particles, mpos, dt,
                                                          pull_strength, particle_speed,
                                                          damp, is_clicked1, is_clicked2,
                                                          CLICK1_STRENGTH, CLICK2_STRENGTH));
         CUDA_CALL(cudaDeviceSynchronize());
         CUDA_CALL(cudaGraphicsUnmapResources(1, &resource));
      }

      /* Draw. */
      GL_CALL(glDrawArrays(GL_POINTS, 0, n_particles));
      glfwSwapBuffers(window);
      glfwPollEvents();

      /* Count FPS. */
      {
         ++frames_drawn;
         float now = glfwGetTime();
         float passed = now - last_fps_time;
         if (passed >= TIME_BETWEEN_FPS_REPORT)
         {
            float fps = frames_drawn / passed;
            std::cout << "\rFPS: " << fps;
            std::cout.flush();
            frames_drawn = 0;
            last_fps_time = now;
         }
      }
   }

   CUDA_CALL(cudaGraphicsUnregisterResource(resource));
   GL_CALL(glDeleteBuffers(1, &vbo));
   GL_CALL(glDeleteVertexArrays(1, &vao));
   glfwTerminate();

   return 0;
}

void glfwErrorCallback(int code, const char* desc)
{
   ERROR("[GLFW Error] '", desc, "' (", code, ")");
}

void windowResizeHandler(GLFWwindow* window, int w, int h)
{
   width = w; height = h;
   GL_CALL(glViewport(0, 0, width, height));
}

void windowKeyInputHandler(GLFWwindow* window, int key, int, int action, int)
{
   switch (key)
   {
      case GLFW_KEY_ESCAPE:
      case GLFW_KEY_Q:
         glfwSetWindowShouldClose(window, GL_TRUE);
         break;
   }
}

__global__
void generateParticles(vec2* pos, vec2* vel, float* imass, const int N,
                       const float imass_min, const float imass_diff,
                       const unsigned long long seed)
{
   size_t idx = threadIdx.x + NPERTHREAD * blockIdx.x * blockDim.x;
   curandState state;
   curand_init(seed, idx, 0, &state);
   for (int i = 0; i < NPERTHREAD && idx < N; ++i)
   {
      float pos_a = 2 * PI * curand_uniform(&state);
      float pos_m = curand_uniform(&state);
      pos[idx] = { pos_m * cos(pos_a), pos_m * sin(pos_a) };
      float vel_a = 2 * PI * curand_uniform(&state);
      float vel_m = curand_uniform(&state);
      vel[idx] = { vel_m * cos(vel_a), vel_m * sin(vel_a) };
      imass[idx] = imass_min + imass_diff * curand_uniform(&state);
      idx += blockDim.x;
   }
}

__global__
void updateParticles(vec2* pos, vec2* vel, float* imass, const int N,
                     vec2 mpos, const float dt, const float pull_strength,
                     const float particle_speed, const float damp,
                     const bool is_clicked1, const bool is_clicked2,
                     const float click1_strength, const float click2_strength)
{
   size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
   if (idx >= N)
      return;

   vec2 p = pos[idx];
   vec2 v = vel[idx];
   float im = imass[idx];

   vec2 f = mpos - p;
   float f_mult = im * pull_strength * dt;
   v += f_mult * f;
   if (is_clicked1)
      v -= (im * click1_strength / magnitude(f)) * f;
   if (is_clicked2)
      v -= (im * click2_strength / length(f)) * f;
   v *= damp;
   
   float v_mult = particle_speed * dt;
   p += v_mult * v;

   pos[idx] = p;
   vel[idx] = v;
}
