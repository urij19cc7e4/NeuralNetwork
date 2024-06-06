#include <iostream>

#include "rfm.h"
#include "cnn.h"
#include "fnn.h"
#include "rnn.h"
#include "nnb.h"
#include "wx_wrapper.h"

using namespace std;

//#include <iomanip>
//#include <fstream>
//
//template <nn_params::nn_activ_t activ>
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

void f()
{
	fnn<nn_params::nn_activ_t::atan> network({ 2, 2, 1 });

	vec<double> input_t[] =
	{
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 }
	};

	vec<double> output_t[] =
	{
		{ -0.5 },
		{ 0.5 },
		{ 0.5 },
		{ -0.5 }
	};

	pipe<info> p;

	wx_wrapper wx;
	wx.create_wnd(p);

	list<info> result = network.train_stoch_mode({ input_t, output_t, 4 }, { nullptr, nullptr, (uint64_t)0 },
		&p, 1000, 75, 25, false, 1, 0.90, 0.00, 0.90, 0.01, 0, 1e-10);

	cout << "0 xor 0 = " << network.pass_fwd({ 0, 0 })(0) << "\n";
	cout << "0 xor 1 = " << network.pass_fwd({ 0, 1 })(0) << "\n";
	cout << "1 xor 0 = " << network.pass_fwd({ 1, 0 })(0) << "\n";
	cout << "1 xor 1 = " << network.pass_fwd({ 1, 1 })(0) << "\n";

	getchar();
}

int main(int argc,char*argv[],char*argp[])
{
	/*to_file<nn_params::nn_activ_t::signed_pos>("signed_pos");
	to_file<nn_params::nn_activ_t::signed_neg>("signed_neg");
	to_file<nn_params::nn_activ_t::thresh_pos>("thresh_pos");
	to_file<nn_params::nn_activ_t::thresh_neg>("thresh_neg");
	to_file<nn_params::nn_activ_t::step_sym>("step_sym");
	to_file<nn_params::nn_activ_t::step_pos>("step_pos");
	to_file<nn_params::nn_activ_t::step_neg>("step_neg");
	to_file<nn_params::nn_activ_t::rad_bas_pos>("rad_bas_pos");
	to_file<nn_params::nn_activ_t::rad_bas_neg>("rad_bas_neg");
	to_file<nn_params::nn_activ_t::sigmoid_log>("sigmoid_log");
	to_file<nn_params::nn_activ_t::sigmoid_rat>("sigmoid_rat");
	to_file<nn_params::nn_activ_t::atan>("atan");
	to_file<nn_params::nn_activ_t::tanh>("tanh");
	to_file<nn_params::nn_activ_t::elu>("elu");
	to_file<nn_params::nn_activ_t::gelu>("gelu");
	to_file<nn_params::nn_activ_t::lelu>("lelu");
	to_file<nn_params::nn_activ_t::relu>("relu");
	to_file<nn_params::nn_activ_t::mish>("mish");
	to_file<nn_params::nn_activ_t::swish>("swish");
	to_file<nn_params::nn_activ_t::softplus>("softplus");*/

	f();

	getchar();
}