#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/glew.h>

#include "Utils/Log.h"
#include "Utils/Error.h"
#include "Graphics/Shader.h"
#include "Math.cuh"
#include "UpdateCUDA.h"
#include "UpdateCPU.h"

/* macros */
#define GET_REQUIRED_ARGUMENT(flag, name) \
   if (strcmp(flag, #name) == 0) \
   { \
      if (i == argc - 1) \
         ERROR("Argument for flag '" #name "' is required."); \
      std::stringstream ss(argv[++i]); \
      ss >> name; \
      continue; \
   }

/* structs, enums */
struct comma_numpunct : public std::numpunct<char>
{
protected:
   char do_thousands_sep() const { return ','; }
   std::string do_grouping() const { return "\03"; }
};

enum class ComputeMode
{
   NONE, CUDA, CPU
};

struct ButtonState
{
   int last_action;
   float last_click_time;
};

/* forward declarations */
static void glfwErrorCallback(int code, const char* desc);
static void windowResizeCallback(GLFWwindow*, int width, int height);
static void keyInputCallback(GLFWwindow* window, int key, int, int action, int);
static void mouseInputCallback(GLFWwindow*, int button, int action, int);
static void updateButtonState(ButtonState* state, int action, bool* is_click_finished,
                              float* click_strength, const float click_strength_max);
static void scrollCallback(GLFWwindow*, double, double yoff);

/* constants */
const char* USAGE_STR =
"Usage: ./OneMillionParticles [OPTION]...\n\n"
"Run particle simulation that follow mouse pointer.\n\n"
"list of possible options:\n"
"   --n_particles             number of particles\n"
"   --cuda                    run simulation using CUDA\n"
"   --cpu                     run simulation using CPU\n"
"   --seed                    random seed\n"
"   --pull_strength           pull force strength\n"
"   --speed_mult              particle speed multiplier\n"
"   --mass_min                minimum particle mass\n"
"   --mass_max                maximum particle mass\n"
"   --damp_factor             particle's velocity damp factor\n"
"   --damp_interval           particle's velocity damp interval\n"
"   --damp_factor_sensitivity damp factor adjustment sensitivity\n"
"   --local_exp_strength      local explosion (mouse1) punch/strength\n"
"   --global_exp_strength     global explosion (mouse2) punch/strength\n"
"   --click_loading_time      time for the explosion to fully load\n"
"   --color_speed_cap         speed threshold for the 'fastest' color\n"
"   --help                    print this help";

static const char* INSTRUCTION_STR =
"Move your mouse cursor to change particles' point of attraction.\n"
"Hold and release left/right mouse button to spawn local/global explosion. The longer you hold, the stronger the explosion.\n"
"Use mouse scollback to control damp factor.\n"
"Press ESCAPE/Q to quit.\n";

static const char* VERTEX_SHADER_SOURCE =
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

static const char* FRAGMENT_SHADER_SOURCE =
R"(
#version 330 core

in vec4 v_Color;
out vec4 f_Color;

void main()
{
   f_Color = v_Color;
}
)";

static constexpr int WINDOW_WIDTH = 1280;
static constexpr int WINDOW_HEIGHT = 720;

static constexpr float TIME_BETWEEN_FPS_REPORT = 2.0f;

static constexpr size_t N_PARTICLES = 1'000'000;
static constexpr unsigned long long SEED = 1234ull;
static constexpr float PULL_STRENGTH = 5.f;
static constexpr float SPEED_MULT = 1.f;
static constexpr float MASS_MIN = 0.5f;
static constexpr float MASS_MAX = 7.0f;
static constexpr float DAMP_FACTOR = 0.995f;
static constexpr float DAMP_INTERVAL = 1.0f / 30;
static constexpr float DAMP_SENSITIVITY = 0.005f;
static constexpr float LOCAL_EXP_STRENGTH_MAX = 10.0f;
static constexpr float GLOBAL_EXP_STRENGTH_MAX = 20.0f;
static constexpr float CLICK_LOADING_TIME = 2.0f;
static constexpr float COLOR_SPEED_CAP = 10.0f;

static constexpr ComputeMode COMPUTE_MODE = ComputeMode::CUDA;

/* variables */
static int width, height;
static float damp_factor;
static float damp_factor_sensitivity;
static float local_exp_strength_max, global_exp_strength_max;
static float click_loading_time;
static bool is_local_exp, is_global_exp;
static float local_exp_strength, global_exp_strength;

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
   damp_factor = DAMP_FACTOR;
   float damp_interval = DAMP_INTERVAL;
   damp_factor_sensitivity = DAMP_SENSITIVITY;
   local_exp_strength_max = LOCAL_EXP_STRENGTH_MAX;
   global_exp_strength_max = GLOBAL_EXP_STRENGTH_MAX;
   click_loading_time = CLICK_LOADING_TIME;
   float color_speed_cap = COLOR_SPEED_CAP;
   ComputeMode compute_mode = ComputeMode::NONE;

   /* Parse program arguments. */
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
      GET_REQUIRED_ARGUMENT(flag, local_exp_strength_max);
      GET_REQUIRED_ARGUMENT(flag, global_exp_strength_max);
      GET_REQUIRED_ARGUMENT(flag, click_loading_time);
      GET_REQUIRED_ARGUMENT(flag, color_speed_cap);
      if (strcmp(flag, "cuda") == 0)
      {
         if (compute_mode != ComputeMode::NONE)
            ERROR("Specified more than one compute mode.");
         compute_mode = ComputeMode::CUDA;
         continue;
      }
      if (strcmp(flag, "cpu") == 0)
      {
         if (compute_mode != ComputeMode::NONE)
            ERROR("Specified more than one compute mode.");
         compute_mode = ComputeMode::CPU;
         continue;
      }
      if (strcmp(flag, "help") == 0)
      {
         print(USAGE_STR);
         exit(EXIT_SUCCESS);
      }
      else
         ERROR("Flag '", arg, "' is no recognized.");
   }
   if (compute_mode == ComputeMode::NONE)
      compute_mode = COMPUTE_MODE;

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
   windowResizeCallback(window, WINDOW_WIDTH, WINDOW_HEIGHT);
   glfwSetFramebufferSizeCallback(window, windowResizeCallback);
   glfwSetKeyCallback(window, keyInputCallback);
   glfwSetMouseButtonCallback(window, mouseInputCallback);
   glfwSetScrollCallback(window, scrollCallback);
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
   
   GLuint vbo;
   GL_CALL(glGenBuffers(1, &vbo));
   GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
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

   /* Initialize simulation. */
   print(INSTRUCTION_STR);
   switch (compute_mode)
   {
      case ComputeMode::CUDA:
         print("CUDA mode selected.");
         CUDA::init(vbo, n_particles, mass_min, mass_max, seed);
         break;
      case ComputeMode::CPU:
         print("CPU mode selected.");
         CPU::init(n_particles, mass_min, mass_max, seed);
         break;
      default:
         assert(false);
   }
   print("Running simulation for ", n_particles, " particles.");

   int frames_drawn = 0;
   float last_loop_time = glfwGetTime();
   float last_fps_time = last_loop_time;
   float last_damp_time = last_loop_time;
   
   while (!glfwWindowShouldClose(window))
   {
      GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

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

      /* Update particles. */
      switch (compute_mode)
      {
         case ComputeMode::CUDA:
            CUDA::updateParticles(n_particles, mouse_pos, dt, pull_strength,
                                  speed_mult, damp, is_local_exp,
                                  is_global_exp, local_exp_strength,
                                  global_exp_strength);
            break;
         case ComputeMode::CPU:
            CPU::updateParticles(n_particles, mouse_pos, dt, pull_strength,
                                 speed_mult, damp, is_local_exp, is_global_exp,
                                 local_exp_strength, global_exp_strength);
            break;
         default:
            assert(false);
      }

      is_local_exp = false;
      is_global_exp = false;

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

   switch (compute_mode)
   {
      case ComputeMode::CUDA:
         CUDA::cleanup();
         break;
      case ComputeMode::CPU:
         break;
      default:
         assert(false);
   }

   GL_CALL(glDeleteProgram(shader));
   GL_CALL(glDeleteBuffers(1, &vbo));
   GL_CALL(glDeleteVertexArrays(1, &vao));
   glfwTerminate();

   return 0;
}

void glfwErrorCallback(int code, const char* desc)
{
   ERROR("[GLFW Error] '", desc, "' (", code, ")");
}

void windowResizeCallback(GLFWwindow* window, int w, int h)
{
   width = w; height = h;
   GL_CALL(glViewport(0, 0, width, height));
}

void keyInputCallback(GLFWwindow* window, int key, int, int action, int)
{
   switch (key)
   {
      case GLFW_KEY_ESCAPE:
      case GLFW_KEY_Q:
         if (action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GL_TRUE);
         break;
   }
}

void mouseInputCallback(GLFWwindow*, int button, int action, int)
{
   static ButtonState left_button_state = { GLFW_RELEASE };
   static ButtonState right_button_state = { GLFW_RELEASE };
   switch (button)
   {
      case GLFW_MOUSE_BUTTON_LEFT:
         updateButtonState(&left_button_state, action, &is_local_exp,
                           &local_exp_strength, local_exp_strength_max);
         break;
      case GLFW_MOUSE_BUTTON_RIGHT:
         updateButtonState(&right_button_state, action, &is_global_exp,
                           &global_exp_strength, global_exp_strength_max);
         break;
   }
}

void updateButtonState(ButtonState* state, int action, bool* is_click_finished,
                       float* click_strength, const float click_strength_max)
{
   if (action != state->last_action)
   {
      float now = glfwGetTime();
      switch (action)
      {
         case GLFW_PRESS:
            state->last_click_time = now;
            break;
         case GLFW_RELEASE:
            float elapsed = now - state->last_click_time;
            float t = std::min(elapsed, click_loading_time) / click_loading_time;
            *click_strength = t * click_strength_max;
            *is_click_finished = true;
            break;
      }
      state->last_action = action;
   }
}

void scrollCallback(GLFWwindow*, double, double yoff)
{
   damp_factor += yoff * damp_factor_sensitivity;
   if (damp_factor > 1) damp_factor = 1;
   else if (damp_factor < 0) damp_factor = 0;
   print("damp_factor=", std::setprecision(3), damp_factor, std::setprecision(2));
}

