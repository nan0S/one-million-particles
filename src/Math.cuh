#pragma once

#include <ostream>

struct vec2
{
   float x;
   float y;
};

__device__ __host__
inline vec2 operator+(vec2 u, vec2 v)
{
   return { u.x + v.x, u.y + v.y };
}

__device__ __host__
inline vec2 operator+=(vec2& u, vec2 v)
{
   u.x += v.x;
   u.y += v.y;
   return u;
}

__device__ __host__
inline vec2 operator-(vec2 u, vec2 v)
{
   return { u.x - v.x, u.y - v.y };
}

__device__ __host__
inline vec2 operator-(vec2 u)
{
   return { -u.x, -u.y };
}

__device__ __host__
inline vec2 operator-=(vec2& u, vec2 v)
{
   u.x -= v.x;
   u.y -= v.y;
   return u;
}

__device__ __host__
inline vec2 operator*(float x, vec2 u)
{
   return { x * u.x, x * u.y };
}

__device__ __host__
inline vec2 operator*(vec2 u, float x)
{
   return x * u;
}

__device__ __host__
inline vec2& operator*=(vec2& u, float x)
{
   u.x *= x;
   u.y *= x;
   return u;
}

__device__ __host__
inline float magnitude(vec2 u)
{
   return u.x * u.x + u.y * u.y;
}

__device__ __host__
inline float length(vec2 u)
{
   return sqrt(magnitude(u));
}

inline std::ostream& operator<<(std::ostream& out, vec2 u)
{
   return out << "(" << u.x << "," << u.y << ")";
}

