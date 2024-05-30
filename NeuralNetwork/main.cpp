#include <iostream>

#include "rfm.h"
#include "cnn.h"
#include "fnn.h"
#include "rnn.h"
#include "nnb.h"

using namespace std;

//#include <iomanip>
//#include <fstream>
//
//template <fnn_params::fnn_activ activ>
//void to_file(string str)
//{
//	ofstream activ_file("C:\\Users\\Urij\\Downloads\\" + str + "_activ.csv");
//	ofstream deriv_file("C:\\Users\\Urij\\Downloads\\" + str + "_deriv.csv");
//
//	activ_file << "X,Y\n";
//	deriv_file << "X,Y\n";
//
//	for (double x = -16; x < 16; x += 0.0625)
//	{
//		activ_file << setw(8) << x << "," << setw(8) << activation<activ>(x, 1) << "\n";
//		deriv_file << setw(8) << x << "," << setw(8) << derivation<activ>(x, activation<activ>(x, 1), 1) << "\n";
//	}
//
//	activ_file.close();
//	deriv_file.close();
//}

int main()
{
	/*to_file<fnn_params::fnn_activ::signed_pos>("signed_pos");
	to_file<fnn_params::fnn_activ::signed_neg>("signed_neg");
	to_file<fnn_params::fnn_activ::thresh_pos>("thresh_pos");
	to_file<fnn_params::fnn_activ::thresh_neg>("thresh_neg");
	to_file<fnn_params::fnn_activ::step_sym>("step_sym");
	to_file<fnn_params::fnn_activ::step_pos>("step_pos");
	to_file<fnn_params::fnn_activ::step_neg>("step_neg");
	to_file<fnn_params::fnn_activ::rad_bas_pos>("rad_bas_pos");
	to_file<fnn_params::fnn_activ::rad_bas_neg>("rad_bas_neg");
	to_file<fnn_params::fnn_activ::sigmoid_log>("sigmoid_log");
	to_file<fnn_params::fnn_activ::sigmoid_rat>("sigmoid_rat");
	to_file<fnn_params::fnn_activ::atan>("atan");
	to_file<fnn_params::fnn_activ::tanh>("tanh");
	to_file<fnn_params::fnn_activ::elu>("elu");
	to_file<fnn_params::fnn_activ::gelu>("gelu");
	to_file<fnn_params::fnn_activ::lelu>("lelu");
	to_file<fnn_params::fnn_activ::relu>("relu");
	to_file<fnn_params::fnn_activ::mish>("mish");
	to_file<fnn_params::fnn_activ::swish>("swish");
	to_file<fnn_params::fnn_activ::softplus>("softplus");*/

	fnn<fnn_params::fnn_activ::tanh, fnn_params::fnn_init_t::normal> network({ 2, 3, 1 });

	vec<double> input_t[] =
	{
		{ 1, 1 },
		{ 0, 0 },
		{ 1, 0 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 1 },
		{ 1, 1 },
		{ 0, 0 }
	};

	vec<double> output_t[] =
	{
		{ 0 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 0 }
	};

	network.train(input_t, output_t, 10, 1000, 1e-6, 0.25);

	cout << "0 xor 0 = " << network.pass_fwd({ 0, 0 })(0) << "\n";
	cout << "0 xor 1 = " << network.pass_fwd({ 0, 1 })(0) << "\n";
	cout << "1 xor 0 = " << network.pass_fwd({ 1, 0 })(0) << "\n";
	cout << "1 xor 1 = " << network.pass_fwd({ 1, 1 })(0) << "\n";

	getchar();
}