#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <cuda_runtime.h>

#include "Utils/Log.h"
#include "Utils/Error.h"
#include "Graphics/Shader.h"
#include "Math.cuh"
#include "UpdateCUDA.h"

/* macros */
#define PI 3.14159265358f
#define NTHREADS 1024
#define NPERTHREAD 1024

#define GET_REQUIRED_ARGUMENT(flag, name) \
   if (strcmp(flag, #name) == 0) \
   { \
      if (i == argc - 1) \
         ERROR("Argument for flag '" #name "' is required."); \
      std::stringstream ss(argv[++i]); \
      ss >> name; \
      continue; \
   }

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

/* constants */
const char* USAGE_STR =
"Usage: ./OneMillionParticles [OPTION]...\n\n"
"Run particle simulation that follow mouse pointer.\n\n"
"list of possible options:\n"
"   --n_particles         number of particles\n"
"   --seed                random seed\n"
"   --pull_strength       pull force strength\n"
"   --speed_mult          particle speed multiplier\n"
"   --mass_min            minimum particle mass\n"
"   --mass_max            maximum particle mass\n"
"   --damp_factor         particle's velocity damp factor\n"
"   --damp_interval       particle's velocity damp interval\n"
"   --local_exp_strength  local explosion (mouse1) punch/strength\n"
"   --global_exp_strength global explosion (mouse2) punch/strength\n"
"   --color_speed_cap     speed threshold for the 'fastest' color\n"
"   --help                print this help";

const char* VERTEX_SHADER_SOURCE =
R"(
#version 330 core

layout (location = 0) in vec2 v_Pos;
layout (location = 1) in vec2 v_Vel;
layout (location = 2) in float v_IMass;

out vec4 v_Color;

uniform float max_speed;
uniform float imax_speed;

void main()
{
   gl_Position = vec4(v_Pos, 0, 1);
   gl_PointSize = (1 / 3.5f) / v_IMass;
   {
      float speed = length(v_Vel);
      float hue;
      if (speed >= max_speed)
         hue = 0;
      else
         hue = (max_speed - speed) * imax_speed * 240;
      float val = hue / 60;
      if (hue < 60)
         v_Color = vec4(1, val, 0, 1);
      else if (hue < 120)
         v_Color = vec4(2 - val, 1, 0, 1);
      else if (hue < 180)
         v_Color = vec4(0, 1, val - 2, 1);
      else
         v_Color = vec4(0, 4 - val, 1, 1);
   }
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
constexpr unsigned long long SEED = 1234ull;
constexpr float PULL_STRENGTH = 5.f;
constexpr float SPEED_MULT = 1.f;
constexpr float MASS_MIN = 0.5f;
constexpr float MASS_MAX = 7.0f;
constexpr float DAMP_FACTOR = 0.995f;
constexpr float DAMP_INTERVAL = 1.0f / 30;
constexpr float LOCAL_EXP_STRENGTH = 1.5f;
constexpr float GLOBAL_EXP_STRENGTH = 3.0f;
constexpr float COLOR_SPEED_CAP = 10.0f;

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
   unsigned long long seed = SEED;
   float pull_strength = PULL_STRENGTH;
   float speed_mult = SPEED_MULT;
   float mass_min = MASS_MIN;
   float mass_max = MASS_MAX;
   float damp_factor = DAMP_FACTOR;
   float damp_interval = DAMP_INTERVAL;
   float local_exp_strength = LOCAL_EXP_STRENGTH;
   float global_exp_strength = GLOBAL_EXP_STRENGTH;
   float color_speed_cap = COLOR_SPEED_CAP;

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
      const char* flag;
      if (arg[1] == '-')
      {
         if (arg[2] == '\0')
            break;
         flag = arg + 2;
      }
      else
         flag = arg + 1;
      GET_REQUIRED_ARGUMENT(flag, n_particles);
      GET_REQUIRED_ARGUMENT(flag, seed);
      GET_REQUIRED_ARGUMENT(flag, pull_strength);
      GET_REQUIRED_ARGUMENT(flag, speed_mult);
      GET_REQUIRED_ARGUMENT(flag, mass_min);
      GET_REQUIRED_ARGUMENT(flag, mass_max);
      GET_REQUIRED_ARGUMENT(flag, damp_factor);
      GET_REQUIRED_ARGUMENT(flag, damp_interval);
      GET_REQUIRED_ARGUMENT(flag, local_exp_strength);
      GET_REQUIRED_ARGUMENT(flag, global_exp_strength);
      GET_REQUIRED_ARGUMENT(flag, color_speed_cap);
      if (strcmp(flag, "help") == 0)
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
   {
      GL_CALL(GLint max_speed_loc = glGetUniformLocation(shader, "max_speed"));
      GL_CALL(glUniform1f(max_speed_loc, COLOR_SPEED_CAP));
      GL_CALL(GLint imax_speed_loc = glGetUniformLocation(shader, "imax_speed"));
      GL_CALL(glUniform1f(imax_speed_loc, 1.0f / COLOR_SPEED_CAP));
   }

   CUDA::init(vbo, n_particles, mass_min, mass_max, seed);

   print("Running simulation for ", n_particles, " particles.");

   int frames_drawn = 0;
   float last_loop_time = glfwGetTime();
   float last_fps_time = last_loop_time;
   float last_damp_time = last_loop_time;
   int last_click1_state = GLFW_RELEASE;
   int last_click2_state = GLFW_RELEASE;
   
   while (!glfwWindowShouldClose(window))
   {
      GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

      /* Handle input. */
      bool is_local_exp, is_global_exp;
      {
         int click1_state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1);
         is_local_exp = click1_state == GLFW_PRESS && last_click1_state == GLFW_RELEASE;
         last_click1_state = click1_state;
         int click2_state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2);
         is_global_exp = click2_state == GLFW_PRESS && last_click2_state == GLFW_RELEASE;
         last_click2_state = click2_state;
      }

      /* Calculate current mouse position. */
      vec2 mouse_pos;
      {
         double x, y;
         glfwGetCursorPos(window, &x, &y);
         mouse_pos.x = 2 * (x - 0.5f * width) / width;
         mouse_pos.y = -2 * (y - 0.5f * height) / height;
      }

      /* Calculate delta time. */
      float dt;
      {
         float now = glfwGetTime();
         dt = now - last_loop_time;
         last_loop_time = now;
      }

      /* Calculate damp. */
      float damp;
      {
         float now = glfwGetTime();
         float passed = now - last_damp_time;
         if (passed >= damp_interval)
         {
            damp = damp_factor;
            last_damp_time = now;
         }
         else
            damp = 1.0f;
      }

      CUDA::updateParticles(n_particles, mouse_pos, dt, pull_strength,
                            speed_mult, damp, is_local_exp, is_global_exp,
                            local_exp_strength, global_exp_strength);

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

