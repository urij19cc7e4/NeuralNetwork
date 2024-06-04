#pragma once

#include <cstdint>
#include <exception>
#include <initializer_list>

constexpr char tns_mtx_sizes_error[] = "Given tensor and matrix are incompatible by size";
constexpr char mtx_vec_sizes_error[] = "Given matrix and vector are incompatible by size";
constexpr char tns_sizes_error[] = "Given tensors are incompatible by size";
constexpr char mtx_sizes_error[] = "Given matrixes are incompatible by size";
constexpr char vec_sizes_error[] = "Given vectors are incompatible by size";

template <typename T, bool initialize = false>
class tns;

template <typename T, bool initialize = false>
class mtx;

template <typename T, bool initialize = false>
class vec;