#include "Error.h"

#include <filesystem>

#include <GL/glew.h>
#include <cuda_runtime.h>

#include "Utils/Log.h"

namespace fs = std::filesystem;

void glClearError()
{
   while (glGetError() != GL_NO_ERROR);
}

bool glLogError(const char* call, const char* file, int line)
{
   GLenum errcode;
   bool good = true;
   const std::string& filename = fs::path(file).filename().string();
   while ((errcode = glGetError()) != GL_NO_ERROR)
   {
      const char* msg = reinterpret_cast<const char*>(gluErrorString(errcode));
      WARNING("[OpenGL Error] ", filename, "::", line, " ", call, " '", msg,
              "' (", errcode, ")");
      good = false;
   }
   return good;
}

void cudaCheckErrors(const char* call, const char* file, int line)
{
   cudaError_t code = cudaGetLastError();
   if (code == cudaSuccess)
      return;
   const std::string& filename = fs::path(file).filename().string();
   const char* err = cudaGetErrorString(code);
   ERROR("CUDA call failure: ", call, " '", err, "' (", code, ") ", "at ",
         filename, ":", line);
}

