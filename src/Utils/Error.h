#pragma once

#include <cassert>

/* macros */
#define GL_CALL(x) \
   glClearError(); \
   x; \
   assert(glLogError(#x, __FILE__, __LINE__));

#define CUDA_CALL(...) \
   __VA_ARGS__; \
   cudaCheckErrors(#__VA_ARGS__, __FILE__, __LINE__)

/* forward declarations */
void glClearError();
bool glLogError(const char* call, const char* file, int line);
void cudaCheckErrors(const char* call, const char* file, int line);

