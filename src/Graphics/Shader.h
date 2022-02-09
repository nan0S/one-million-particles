#pragma once

#include <GL/glew.h>

namespace Graphics
{
   GLuint createGraphicsShader(const char* vertex_source,
                               const char* fragment_source);
   GLuint createComputeShader(const char* compute_source);
}
