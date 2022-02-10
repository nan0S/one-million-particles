#pragma once

#include <GL/glew.h>

#include "Simulation.h"

namespace Compute
{
   void init(const SimConfig* sim_config, GLuint vbo, GLuint render_shader);
   void updateParticles(const SimUpdate* sim_update, GLuint vbo,
                        GLuint render_shader);
   void cleanup();
}
