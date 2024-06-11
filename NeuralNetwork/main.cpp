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

void f(wx_wrapper&wx)
{
	fnn network({ 2, 3, 1 },nn_params::nn_activ_t::atan);

	vec<double> input_t[] =
	{
		{ -1, -1 },
		{ -1, 1 },
		{ 1, -1 },
		{ 1, 1 }
	};

	vec<double> output_t[] =
	{
		{ -0.5 },
		{ 0.5 },
		{ 0.5 },
		{ -0.5 }
	};

	vec<double> input_test[] =
	{
		{ -2, 0 },
		{ 5, 1 },
		{ -1, 0 },
		{ -2, -2 }
	};

	vec<double> output_test[] =
	{
		{ 0.5 },
		{ 0.5 },
		{ 0.5 },
		{ -0.5 }
	};

	pipe<info>* p = new pipe<info>();
	pipe<info>* pp = new pipe<info>();

	string s("Fuck");
	wx.create_graph_wnd(*pp,s);
	for (double x = 0.0, z = 0.0; x <= 10.0; x += 0.01, z += 0.01)
	{
		pp->push(info(sin(x), cos(z)));
		this_thread::sleep_for(chrono::microseconds(10));
	}

	network.train_stoch_mode({ input_t, output_t, 4 }, { nullptr, nullptr, 0 },nullptr,
		p, 1000, 75, 25, false, 1, 0.90, 0.10, 0.90, 0.25, 1000, 1e-10);

	cout << "0 xor 0 = " << network.pass_fwd({ -1, -1 })(0) << "\n";
	cout << "0 xor 1 = " << network.pass_fwd({ -1, 1 })(0) << "\n";
	cout << "1 xor 0 = " << network.pass_fwd({ 1, -1 })(0) << "\n";
	cout << "1 xor 1 = " << network.pass_fwd({ 1, 1 })(0) << "\n";
	cout << "\n";
	cout << "-2 xor 0 = " << network.pass_fwd({ -2, 0 })(0) << "\n";
	cout << "5 xor 1 = " << network.pass_fwd({ 5, 1 })(0) << "\n";
	cout << "-1 xor 0 = " << network.pass_fwd({ -1, 0 })(0) << "\n";
	cout << "-2 xor -2 = " << network.pass_fwd({ -2, -2 })(0) << "\n";

	pp->push(info(msg_type::max_epo_reached));
	this_thread::sleep_for(chrono::milliseconds(1000));
	delete p;
	delete pp;
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
	/*{
		wx_wrapper wx;
		for (int i = 0; i < 1; ++i)
			f(wx);
	}*/

	tns<double> tensor(
		{
			{
				{ 20.0, 18.0 },
				{ 22.0, 10.0 },
				{ 30.0, 17.0 },
				{ 44.0, 60.0 },
				{ 25.0, 50.0 }
			},
			{
				{ 10.0, 19.0 },
				{ 20.0, 58.0 },
				{ 15.0, 16.0 },
				{ 40.0, 60.0 },
				{ 50.0, 55.0 }
			},
			{
				{ 1.0, 9.0 },
				{ 2.0, 8.0 },
				{ 3.0, 7.0 },
				{ 4.0, 6.0 },
				{ 5.0, 5.0 }
			},
			{
				{ 1.0, 9.0 },
				{ 2.0, 8.0 },
				{ 3.0, 7.0 },
				{ 4.0, 6.0 },
				{ 5.0, 5.0 }
			},
			{
				{ 1.0, 9.0 },
				{ 2.0, 8.0 },
				{ 3.0, 7.0 },
				{ 4.0, 6.0 },
				{ 5.0, 5.0 }
			}
		}
	);

	tns<double> core(
		{
			{
				{ -1.0, -1.0, -1.0 },
				{ -1.0, 10.0, -1.0 },
				{ -1.0, -1.0, -1.0 },
			}
		}
	);

	tns<double> result = arithmetic::convolute(tensor, core, vec<double>({ 0.0 }));

	getchar();
}