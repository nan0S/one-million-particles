#pragma once

#include <ostream>
#include <cmath>

#include <cuda_runtime.h>

struct vec2
{
   float x;
   float y;
};

__host__ __device__
inline vec2 operator+(vec2 u, vec2 v)
{
   return { u.x + v.x, u.y + v.y };
}

__host__ __device__
inline vec2 operator+=(vec2& u, vec2 v)
{
   u.x += v.x;
   u.y += v.y;
   return u;
}

__host__ __device__
inline vec2 operator-(vec2 u, vec2 v)
{
   return { u.x - v.x, u.y - v.y };
}

__host__ __device__
inline vec2 operator-(vec2 u)
{
   return { -u.x, -u.y };
}

__host__ __device__
inline vec2 operator-=(vec2& u, vec2 v)
{
   u.x -= v.x;
   u.y -= v.y;
   return u;
}

__host__ __device__
inline vec2 operator*(float x, vec2 u)
{
   return { x * u.x, x * u.y };
}

__host__ __device__
inline vec2 operator*(vec2 u, float x)
{
   return x * u;
}

__host__ __device__
inline vec2& operator*=(vec2& u, float x)
{
   u.x *= x;
   u.y *= x;
   return u;
}

__host__ __device__
inline float magnitude(vec2 u)
{
   return u.x * u.x + u.y * u.y;
}

__host__ __device__
inline float length(vec2 u)
{
   return std::sqrt(magnitude(u));
}

inline std::ostream& operator<<(std::ostream& out, vec2 u)
{
   return out << "(" << u.x << "," << u.y << ")";
}

