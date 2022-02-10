#pragma once

#include <GL/glew.h>

#include "Simulation.h"

namespace CUDA
{
   void init(const SimConfig* sim_config, GLuint vbo);
   void updateParticles(const SimUpdate* sim_update);
   void cleanup();
}
