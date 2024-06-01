#pragma once

#include <cstdint>
#include <exception>
#include <initializer_list>

constexpr char mtx_vec_sizes_error[] = "Given matrix and vector are incompatible by size";
constexpr char mtx_sizes_error[] = "Given matrixes are incompatible by size";
constexpr char vec_sizes_error[] = "Given vectors are incompatible by size";

template <typename T, bool initialize = false>
class mtx;

template <typename T, bool initialize = false>
class vec;