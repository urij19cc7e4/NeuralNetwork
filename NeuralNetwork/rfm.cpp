#include "rfm.h"

extern template vec<long double> arithmetic::apply_multiply(
	const mtx<long double>& _mtx, const vec<long double>& _vec,
	function<long double(long double)> apply);
extern template vec<double> arithmetic::apply_multiply(
	const mtx<double>& _mtx, const vec<double>& _vec,
	function<double(double)> apply);
extern template vec<float> arithmetic::apply_multiply(
	const mtx<float>& _mtx, const vec<float>& _vec,
	function<float(float)> apply);

extern template vec<long double> arithmetic::apply_multiply(
	const vec<long double>& _vec, const mtx<long double>& _mtx,
	function<long double(long double)> apply);
extern template vec<double> arithmetic::apply_multiply(
	const vec<double>& _vec, const mtx<double>& _mtx,
	function<double(double)> apply);
extern template vec<float> arithmetic::apply_multiply(
	const vec<float>& _vec, const mtx<float>& _mtx,
	function<float(float)> apply);

extern template tuple<vec<long double>, vec<long double>> arithmetic::apply_multiply(
	const mtx<long double>& _mtx, const vec<long double>& _vec,
	function<long double(long double)> apply,
	function<long double(long double, long double)> again);
extern template tuple<vec<double>, vec<double>> arithmetic::apply_multiply(
	const mtx<double>& _mtx, const vec<double>& _vec,
	function<double(double)> apply,
	function<double(double, double)> again);
extern template tuple<vec<float>, vec<float>> arithmetic::apply_multiply(
	const mtx<float>& _mtx, const vec<float>& _vec,
	function<float(float)> apply,
	function<float(float, float)> again);

extern template tuple<vec<long double>, vec<long double>> arithmetic::apply_multiply(
	const vec<long double>& _vec, const mtx<long double>& _mtx,
	function<long double(long double)> apply,
	function<long double(long double, long double)> again);
extern template tuple<vec<double>, vec<double>> arithmetic::apply_multiply(
	const vec<double>& _vec, const mtx<double>& _mtx,
	function<double(double)> apply,
	function<double(double, double)> again);
extern template tuple<vec<float>, vec<float>> arithmetic::apply_multiply(
	const vec<float>& _vec, const mtx<float>& _mtx,
	function<float(float)> apply,
	function<float(float, float)> again);

extern template vec<long double> arithmetic::operator*(const mtx<long double>& _mtx, const vec<long double>& _vec);
extern template vec<double> arithmetic::operator*(const mtx<double>& _mtx, const vec<double>& _vec);
extern template vec<float> arithmetic::operator*(const mtx<float>& _mtx, const vec<float>& _vec);

extern template vec<long double> arithmetic::operator*(const vec<long double>& _vec, const mtx<long double>& _mtx);
extern template vec<double> arithmetic::operator*(const vec<double>& _vec, const mtx<double>& _mtx);
extern template vec<float> arithmetic::operator*(const vec<float>& _vec, const mtx<float>& _mtx);