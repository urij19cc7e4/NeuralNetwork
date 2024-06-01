#include "rfm.h"

extern template vec<long double> arithmetic::operator*(const mtx<long double>& _mtx, const vec<long double>& _vec);
extern template vec<double> arithmetic::operator*(const mtx<double>& _mtx, const vec<double>& _vec);
extern template vec<float> arithmetic::operator*(const mtx<float>& _mtx, const vec<float>& _vec);

extern template vec<long double> arithmetic::operator*(const vec<long double>& _vec, const mtx<long double>& _mtx);
extern template vec<double> arithmetic::operator*(const vec<double>& _vec, const mtx<double>& _mtx);
extern template vec<float> arithmetic::operator*(const vec<float>& _vec, const mtx<float>& _mtx);