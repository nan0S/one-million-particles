#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "Utils/Log.h"
#include "Utils/Error.h"
#include "Utils/Timer.h"
#include "Graphics/Shader.h"
#include "Simulation.h"
#include "UpdateCUDA.h"
#include "UpdateCPU.h"
#include "UpdateCompute.h"

/* macros */
#define GET_REQUIRED_ARGUMENT(argstr, var) \
   if (strcmp(argstr, flag) == 0) \
   { \
      if (i == argc - 1) \
         ERROR("Argument for flag '" argstr "' is required."); \
      std::stringstream ss(argv[++i]); \
      ss >> var; \
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
   NONE, CUDA, CPU, COMPUTE
};

struct ButtonState
{
   int last_action;
   float last_click_time;
};

struct SimParams
{
   float damp;
   float damp_interval;
   float local_exp_strength_max;
   float global_exp_strength_max;
   float click_loading_time;
   float color_speed_cap;
};

struct WindowContext
{
   int width;
   int height;
   SimParams* sim_params;
   SimUpdate* sim_update;
};

/* forward declarations */
static void resetParamsToDefault(SimParams* sim_params, SimUpdate* sim_update);
static void glfwErrorCallback(int code, const char* desc);
static void windowResizeCallback(GLFWwindow*, int width, int height);
static void keyInputCallback(GLFWwindow* window, int key, int, int action, int);
static void mouseInputCallback(GLFWwindow*, int button, int action, int);
static void updateButtonState(ButtonState* state, int action, bool* is_click_finished,
                              float* click_strength, const float click_strength_max,
                              const float click_loading_time);

/* constants */
const char* USAGE_STR =
"Usage: ./OneMillionParticles [OPTION]...\n\n"
"Run particle simulation that follow mouse pointer.\n\n"
"list of possible options:\n"
"   --n_particles             number of particles (= 1000000)\n"
"   --cuda                    run simulation using CUDA (DEFAULT)\n"
"   --cpu                     run simulation using CPU\n"
"   --compute                 run simulation using Compute Shader\n"
"   --seed                    random seed (= 1234)\n"
"   --mass_min                minimum particle mass (= 0.5)\n"
"   --mass_max                maximum particle mass (= 7.0)\n"
"   --pull_strength           pull force strength (= 5.0)\n"
"   --speed_mult              particle speed multiplier (= 1.0)\n"
"   --damp                    particle's velocity damp factor (= 0.995)\n"
"   --damp_interval           particle's velocity damp interval (= 0.033)\n"
"   --local_exp_strength_max  local explosion (mouse1) maximum punch/strength (= 10.0)\n"
"   --global_exp_strength_max global explosion (mouse2) maximum punch/strength (= 20.0)\n"
"   --click_loading_time      time for the explosion to fully load (= 2.0)\n"
"   --color_speed_cap         speed threshold for the 'fastest' color (= 10.0)\n"
"   --help                    print this help";

static const char* INSTRUCTION_STR =
"Move your mouse cursor to change particles' point of attraction.\n"
"Hold and release left/right mouse button to spawn local/global explosion. The longer you hold, the stronger the explosion.\n"
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
static constexpr size_t N_PARTICLES = 1'000'000;
static constexpr unsigned long long SEED = 1234ull;
static constexpr float PULL_STRENGTH = 5.f;
static constexpr float SPEED_MULT = 1.f;
static constexpr float MASS_MIN = 0.5f;
static constexpr float MASS_MAX = 7.0f;
static constexpr float DAMP = 0.995f;
static constexpr float DAMP_INTERVAL = 1.0f / 30;
static constexpr float LOCAL_EXP_STRENGTH_MAX = 10.0f;
static constexpr float GLOBAL_EXP_STRENGTH_MAX = 20.0f;
static constexpr float CLICK_LOADING_TIME = 2.0f;
static constexpr float COLOR_SPEED_CAP = 10.0f;
static constexpr ComputeMode COMPUTE_MODE = ComputeMode::CUDA;

int main(int argc, char* argv[])
{
   {
      std::locale comma_locale(std::locale(), new comma_numpunct);
      std::cout.imbue(comma_locale);
      std::cerr.imbue(comma_locale);
      std::cout << std::setprecision(2) << std::fixed;
   }

   /* Initialize simulation params with default values. */
   SimConfig sim_config = {};
   sim_config.n_particles = N_PARTICLES;
   sim_config.seed = SEED;
   sim_config.mass_min = MASS_MIN;
   sim_config.mass_max = MASS_MAX;
   SimParams sim_params = {};
   SimUpdate sim_update = {};
   resetParamsToDefault(&sim_params, &sim_update);
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
      GET_REQUIRED_ARGUMENT("n_particles", sim_config.n_particles);
      GET_REQUIRED_ARGUMENT("seed", sim_config.seed);
      GET_REQUIRED_ARGUMENT("mass_min", sim_config.mass_min);
      GET_REQUIRED_ARGUMENT("mass_max", sim_config.mass_max);
      GET_REQUIRED_ARGUMENT("pull_strength", sim_update.pull_strength);
      GET_REQUIRED_ARGUMENT("speed_mult", sim_update.speed_mult);
      GET_REQUIRED_ARGUMENT("damp", sim_params.damp);
      GET_REQUIRED_ARGUMENT("damp_interval", sim_params.damp_interval);
      GET_REQUIRED_ARGUMENT("local_exp_strength_max", sim_params.local_exp_strength_max);
      GET_REQUIRED_ARGUMENT("global_exp_strength_max", sim_params.global_exp_strength_max);
      GET_REQUIRED_ARGUMENT("click_loading_time", sim_params.click_loading_time);
      GET_REQUIRED_ARGUMENT("color_speed_cap", sim_params.color_speed_cap);
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
      if (strcmp(flag, "compute") == 0)
      {
         if (compute_mode != ComputeMode::NONE)
            ERROR("Specified more than one compute mode.");
         compute_mode = ComputeMode::COMPUTE;
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
   
   WindowContext window_context;
   window_context.sim_params = &sim_params;
   window_context.sim_update = &sim_update;
   glfwMakeContextCurrent(window);
   glfwSetWindowUserPointer(window, (void*)&window_context);
   windowResizeCallback(window, WINDOW_WIDTH, WINDOW_HEIGHT);
   glfwSetFramebufferSizeCallback(window, windowResizeCallback);
   glfwSetKeyCallback(window, keyInputCallback);
   glfwSetMouseButtonCallback(window, mouseInputCallback);
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
                                reinterpret_cast<const void*>(sim_config.n_particles *
                                                              sizeof(vec2))));
   GL_CALL(glEnableVertexAttribArray(2));
   GL_CALL(glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE,
                                sizeof(float),
                                reinterpret_cast<const void*>(2 * sim_config.n_particles *
                                                              sizeof(vec2))));

   GLuint shader = Graphics::createGraphicsShader(VERTEX_SHADER_SOURCE,
                                                  FRAGMENT_SHADER_SOURCE);
   GL_CALL(glUseProgram(shader));
   GL_CALL(GLint max_speed_loc = glGetUniformLocation(shader, "max_speed"));
   GL_CALL(GLint imax_speed_loc = glGetUniformLocation(shader, "imax_speed"));
   GL_CALL(glUniform1f(max_speed_loc, sim_params.color_speed_cap));
   GL_CALL(glUniform1f(imax_speed_loc, 1.0f / sim_params.color_speed_cap));

   /* Setup ImGui. */
   {
      ImGui::CreateContext();
      ImGuiIO& io = ImGui::GetIO();
      io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
      io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
      io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
      ImGui::StyleColorsDark();
      ImGuiStyle& style = ImGui::GetStyle();
      style.WindowRounding = 0.0f;
      style.Colors[ImGuiCol_WindowBg].w = 1.0f;
      ImGui_ImplGlfw_InitForOpenGL(window, true);
      ImGui_ImplOpenGL3_Init("#version 330");
   }

   /* Initialize simulation. */
   print(INSTRUCTION_STR);
   switch (compute_mode)
   {
      case ComputeMode::CUDA:
         print("CUDA mode selected.");
         CUDA::init(&sim_config, vbo);
         break;
      case ComputeMode::CPU:
         print("CPU mode selected.");
         CPU::init(&sim_config);
         break;
      case ComputeMode::COMPUTE:
         print("COMPUTE mode selected.");
         Compute::init(&sim_config, vbo, shader);
         break;
      default:
         assert(false);
   }
   print("Running simulation for ", sim_config.n_particles, " particles.");

   float last_loop_time = glfwGetTime();
   float last_damp_time = last_loop_time;
   
   while (!glfwWindowShouldClose(window))
   {
      GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

      /* Calculate current mouse position. */
      {
         double x, y;
         glfwGetCursorPos(window, &x, &y);
         sim_update.mouse_pos.x = 2 * (x - 0.5f * window_context.width) /
            window_context.width;
         sim_update.mouse_pos.y = -2 * (y - 0.5f * window_context.height) /
            window_context.height;
      }
      /* Calculate delta time. */
      {
         float now = glfwGetTime();
         sim_update.delta_time = now - last_loop_time;
         last_loop_time = now;
      }
      /* Calculate damp. */
      {
         float now = glfwGetTime();
         float passed = now - last_damp_time;
         if (passed >= sim_params.damp_interval)
         {
            sim_update.damp = sim_params.damp;
            last_damp_time = now;
         }
         else
            sim_update.damp = 1.0f;
      }

      /* Update particles. */
      switch (compute_mode)
      {
         case ComputeMode::CUDA:
            CUDA::updateParticles(&sim_update);
            break;
         case ComputeMode::CPU:
            CPU::updateParticles(&sim_update);
            break;
         case ComputeMode::COMPUTE:
            Compute::updateParticles(&sim_update, vbo, shader);
            break;
         default:
            assert(false);
      }

      sim_update.is_local_exp = false;
      sim_update.is_global_exp = false;

      /* Draw particles. */
      GL_CALL(glDrawArrays(GL_POINTS, 0, sim_config.n_particles));

      /* Draw ImGui. */
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      ImGui::Begin("Simulation Parameters");
      ImGui::SliderFloat("Pull Strength", &sim_update.pull_strength, 0.0f, 50.0f);
      ImGui::SliderFloat("Speed Mult", &sim_update.speed_mult, 0.0f, 10.0f);
      ImGui::SliderFloat("Damp", &sim_params.damp, 0.0f, 1.0f);
      ImGui::SliderFloat("Damp Interval", &sim_params.damp_interval, 0.0f, 0.5f);
      ImGui::SliderFloat("Local Exp Strength Max", &sim_params.local_exp_strength_max, 0.0f, 50.0f);
      ImGui::SliderFloat("Global Exp Strength Max", &sim_params.global_exp_strength_max, 0.0f, 100.0f);
      ImGui::SliderFloat("Click Loading Time", &sim_params.click_loading_time, 0.005f, 10.0f);
      if (ImGui::SliderFloat("Color Speed Cap", &sim_params.color_speed_cap, 0.0f, 100.0f))
      {
         GL_CALL(glUniform1f(max_speed_loc, sim_params.color_speed_cap));
         GL_CALL(glUniform1f(imax_speed_loc, 1.0f / sim_params.color_speed_cap));
      }
      if (ImGui::Button("Reset to defaults"))
      {
         resetParamsToDefault(&sim_params, &sim_update);
         GL_CALL(glUniform1f(max_speed_loc, sim_params.color_speed_cap));
         GL_CALL(glUniform1f(imax_speed_loc, 1.0f / sim_params.color_speed_cap));
      }
      ImGui::Text("Simulation average %.2f ms/frame (%.1f FPS)",
            1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGui::End();
      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      GLFWwindow* backup_current_context = glfwGetCurrentContext();
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
      glfwMakeContextCurrent(backup_current_context);

      /* Display. */
      glfwSwapBuffers(window);
      glfwPollEvents();
   }

   switch (compute_mode)
   {
      case ComputeMode::CUDA:
         CUDA::cleanup();
         break;
      case ComputeMode::CPU:
         break;
      case ComputeMode::COMPUTE:
         Compute::cleanup();
         break;
      default:
         assert(false);
   }

   ImGui_ImplOpenGL3_Shutdown();
   ImGui_ImplGlfw_Shutdown();
   ImGui::DestroyContext();
   GL_CALL(glDeleteProgram(shader));
   GL_CALL(glDeleteBuffers(1, &vbo));
   GL_CALL(glDeleteVertexArrays(1, &vao));
   glfwTerminate();

   return 0;
}

void resetParamsToDefault(SimParams* sim_params, SimUpdate* sim_update)
{
   sim_update->pull_strength = PULL_STRENGTH;
   sim_update->speed_mult = SPEED_MULT;
   sim_params->damp = DAMP;
   sim_params->damp_interval = DAMP_INTERVAL;
   sim_params->local_exp_strength_max = LOCAL_EXP_STRENGTH_MAX;
   sim_params->global_exp_strength_max = GLOBAL_EXP_STRENGTH_MAX;
   sim_params->click_loading_time = CLICK_LOADING_TIME;
   sim_params->color_speed_cap = COLOR_SPEED_CAP;
}

void glfwErrorCallback(int code, const char* desc)
{
   ERROR("[GLFW Error] '", desc, "' (", code, ")");
}

void windowResizeCallback(GLFWwindow* window, int width, int height)
{
   WindowContext* context = (WindowContext*)glfwGetWindowUserPointer(window);
   context->width = width;
   context->height = height;
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

void mouseInputCallback(GLFWwindow* window, int button, int action, int)
{
   static ButtonState left_button_state = { GLFW_RELEASE };
   static ButtonState right_button_state = { GLFW_RELEASE };
   WindowContext* context = (WindowContext*)glfwGetWindowUserPointer(window);
   switch (button)
   {
      case GLFW_MOUSE_BUTTON_LEFT:
         updateButtonState(&left_button_state, action,
                           &context->sim_update->is_local_exp,
                           &context->sim_update->local_exp_strength,
                           context->sim_params->local_exp_strength_max,
                           context->sim_params->click_loading_time);
         break;
      case GLFW_MOUSE_BUTTON_RIGHT:
         updateButtonState(&right_button_state, action,
                           &context->sim_update->is_global_exp,
                           &context->sim_update->global_exp_strength,
                           context->sim_params->global_exp_strength_max,
                           context->sim_params->click_loading_time);
         break;
   }
}

void updateButtonState(ButtonState* state, int action, bool* is_click_finished,
                       float* click_strength, const float click_strength_max,
                       const float click_loading_time)
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

