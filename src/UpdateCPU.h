#pragma once

#include <GL/glew.h>

#include "Simulation.h"

namespace CPU
{
   void init(const SimConfig* sim_config);
   void updateParticles(const SimUpdate* sim_update);
}
