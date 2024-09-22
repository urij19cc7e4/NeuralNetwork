#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "nn_rfm.h"
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

//void f(wx_wrapper&wx)
//{
//	fnn_train network({ 2, 3, 1 },nn_params::nn_activ_t::atan);
//
//	vec<double> input_t[] =
//	{
//		{ -1, -1 },
//		{ -1, 1 },
//		{ 1, -1 },
//		{ 1, 1 }
//	};
//
//	vec<double> output_t[] =
//	{
//		{ -0.5 },
//		{ 0.5 },
//		{ 0.5 },
//		{ -0.5 }
//	};
//
//	vec<double> input_test[] =
//	{
//		{ -2, 0 },
//		{ 5, 1 },
//		{ -1, 0 },
//		{ -2, -2 }
//	};
//
//	vec<double> output_test[] =
//	{
//		{ 0.5 },
//		{ 0.5 },
//		{ 0.5 },
//		{ -0.5 }
//	};
//
//	pipe<info>* p = new pipe<info>();
//
//	wx.create_graph_wnd(*p);
//
//	network.train_stoch_mode({ input_t, output_t, 4 }, { nullptr, nullptr, 0 },nullptr,
//		p, 1000, 75, 25, false, 1, 0.90, 0.10, 0.90, 0.25, 1000, 1e-10);
//
//	cout << "0 xor 0 = " << network.pass_fwd({ -1, -1 })(0) << "\n";
//	cout << "0 xor 1 = " << network.pass_fwd({ -1, 1 })(0) << "\n";
//	cout << "1 xor 0 = " << network.pass_fwd({ 1, -1 })(0) << "\n";
//	cout << "1 xor 1 = " << network.pass_fwd({ 1, 1 })(0) << "\n";
//	cout << "\n";
//	cout << "-2 xor 0 = " << network.pass_fwd({ -2, 0 })(0) << "\n";
//	cout << "5 xor 1 = " << network.pass_fwd({ 5, 1 })(0) << "\n";
//	cout << "-1 xor 0 = " << network.pass_fwd({ -1, 0 })(0) << "\n";
//	cout << "-2 xor -2 = " << network.pass_fwd({ -2, -2 })(0) << "\n";
//
//	this_thread::sleep_for(chrono::milliseconds(1000));
//	delete p;
//}

#include "light_appx.h"

void read_mnist(vector<shared_ptr<::data<FLT>>>& idata, vector<shared_ptr<::data<FLT>>>& odata, uint64_t count, string path)
{
	ifstream data_file(path+"images.idx3-ubyte", ios::binary);
	ifstream label_file(path+"labels.idx1-ubyte", ios::binary);

	int32_t _data, _label;

	data_file.read((char*)&_data, sizeof(_data));
	label_file.read((char*)&_label, sizeof(_label));

	if (_byteswap_ulong(_data) == 0x0803 && _byteswap_ulong(_label) == 0x0801)
	{
		uint64_t height, width;

		data_file.read((char*)&_data, sizeof(_data));
		label_file.read((char*)&_label, sizeof(_label));

		_data = _byteswap_ulong(_data);
		_label = _byteswap_ulong(_label);

		if (count < _data)
			count = _data;

		if (count < _label)
			count = _label;

		data_file.read((char*)&_data, sizeof(_data));
		height = (uint64_t)_byteswap_ulong(_data);
		data_file.read((char*)&_data, sizeof(_data));
		width = (uint64_t)_byteswap_ulong(_data);

		for (uint64_t i = (uint64_t)0; i < count; ++i)
		{
			idata.push_back(move((shared_ptr<::data<FLT>>)(new tns<FLT>((uint64_t)1, height+4, width+4))));
			odata.push_back(move((shared_ptr<::data<FLT>>)(new vec<FLT>((uint64_t)10))));

			for (uint64_t j = (uint64_t)0; j < idata[i]->get_size(); ++j)
			{
				if(j/(height+4)>=2&&j/(height+4)<30&&j%(width+4)>=2&&j%(width+4)<30)
				{
				uint8_t data_byte=0;
				data_file.read((char*)&data_byte, sizeof(data_byte));
				(*(idata[i]))(j) = (FLT)data_byte/(FLT)255;
				}
				else
				(*(idata[i]))(j) = (FLT)0;
			}

			for (uint64_t j = (uint64_t)0; j < odata[i]->get_size(); ++j)
				(*(odata[i]))(j) = (FLT)-0.1;

			uint8_t label_byte;
			label_file.read((char*)&label_byte, sizeof(label_byte));
			(*(odata[i]))((uint64_t)label_byte) = (FLT)0.9;
		}
	}
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

	/*pipe<info> pip;
	wx_wrapper wx;
	wx.create_graph_wnd(pip);
	light_appx l(1000, 0, 10000, 0, 10);
	for (int i=0;i<10000;++i)
	{
		pip.push(info(l.forward(),0));
		this_thread::sleep_for(chrono::microseconds(1));
	}*/

	data_set train_data, test_data;

	train_data.idata.reserve(60000);
	train_data.odata.reserve(60000);
	train_data.size = 60000;

	test_data.idata.reserve(10000);
	test_data.odata.reserve(10000);
	test_data.size = 10000;

	read_mnist(train_data.idata, train_data.odata, train_data.size, R"(C:\Users\Urij\Downloads\train-)");
	read_mnist(test_data.idata, test_data.odata, test_data.size, R"(C:\Users\Urij\Downloads\t10k-)");

	pipe<info> pip;
	wx_wrapper wx;
	wx.create_graph_wnd(pip);

	/*nnb network(
		{
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(2,1,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, false)),
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(4,2,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, false)),
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(8,4,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, false)),
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(16,8,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, false)),
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(32,16,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, true)),
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(64,32,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, false)),
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(128,64,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, false)),
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(256,128,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, false)),
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(512,256,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, false)),

			unique_ptr<nn_info>((nn_info*)new cnn_2_fnn_info(true,false)),

			unique_ptr<nn_info>((nn_info*)new fnn_info
				(512, 1096, nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1)),
			unique_ptr<nn_info>((nn_info*)new fnn_info
				(1096, 96, nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1)),
			unique_ptr<nn_info>((nn_info*)new fnn_info
				(96, 10, nn_params::nn_activ_t::tanh, nn_params::nn_init_t::normal, 1, 1, 1))
		}
	);*/

	nnb network(
		{
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(8,1,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, true)),
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(16,8,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, true)),
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(20,16,3, 3,nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal, 1, 1, 1, true)),

			unique_ptr<nn_info>((nn_info*)new cnn_2_fnn_info(true,true)),

			unique_ptr<nn_info>((nn_info*)new fnn_info
				(320, 640, nn_params::nn_activ_t::tanh, nn_params::nn_init_t::normal, 1, 1, 1)),
			unique_ptr<nn_info>((nn_info*)new fnn_info
				(640, 120, nn_params::nn_activ_t::tanh, nn_params::nn_init_t::normal, 1, 1, 1)),
			unique_ptr<nn_info>((nn_info*)new fnn_info
				(120, 24, nn_params::nn_activ_t::tanh, nn_params::nn_init_t::normal, 1, 1, 1)),
			unique_ptr<nn_info>((nn_info*)new fnn_info
				(24, 10, nn_params::nn_activ_t::tanh, nn_params::nn_init_t::normal, 1, 1, 1))
		}
	);

	list<phase> phase_list;

	phase_list.push_back(move(phase(train_mode::MINI_BATCH | train_mode::CROSS_TEST | train_mode::DROP_OUT,
		32, 25, 1050, 0.02, 8, 0.975, 0.975, 0.00000025, 0.00000025, 25.0, 1e-9)));
	phase_list.push_back(move(phase(train_mode::MINI_BATCH | train_mode::CROSS_TEST | train_mode::DROP_OUT,
		16, 25, 1450, 0.02, 8, 0.975, 0.975, 0.0000005, 0.0000005, 25.0, 1e-9)));
	phase_list.push_back(move(phase(train_mode::MINI_BATCH | train_mode::CROSS_TEST | train_mode::DROP_OUT,
		8, 25, 25000, 0.01, 16, 0.975, 0.975, 0.0000095, 0.0000025, 25.0, 1e-9)));
	phase_list.push_back(move(phase(train_mode::MINI_BATCH | train_mode::CROSS_TEST,
		4, 50, 15000, 0.00, 10, 0.925, 0.925, 0.0000025, 0.0000015, 75.0, 1e-9)));
	phase_list.push_back(move(phase(train_mode::STOCHASTIC | train_mode::CROSS_TEST,
		2, 75, 7500, 0.00, 10, 0.925, 0.925, 0.0000015, 0.0000005, 975.0, 1e-9)));

	cout << network.get_param_count();
	network.train(train_data, test_data, train_mode::NONE, &phase_list, 0, &pip);

	ofstream my_file("my_network_2.dat", ios_base::binary);
	network.save_to_file(my_file);
	my_file.close();

	uint64_t counter = 0;
	for (int i = 0; i < 10000; ++i)
	{
		vec<FLT>* result = (vec<FLT>*)network.pass(*(test_data.idata[i]));

		FLT max = (*result)(0);
		int max_i = 0;

		for (int j = 1; j < 10; ++j)
			if (max < (*result)(j))
			{
				max = (*result)(j);
				max_i = j;
			}

		if ((*(test_data.odata[i]))(max_i) == (FLT)0.9)
			++counter;
	}

	cout <<"\n" << (FLT)counter / 10000.0;

	/*unique_ptr<unique_ptr<::data<FLT>>[]> i_data((unique_ptr<::data<FLT>>*)new unique_ptr<tns<FLT>>[2]());
	unique_ptr<unique_ptr<::data<FLT>>[]> o_data((unique_ptr<::data<FLT>>*)new unique_ptr<tns<FLT>>[2]());

	i_data[0] = unique_ptr<::data<FLT>>(new tns<FLT>(
		{
			{
				{ 0.0, 1.0, 0.0 },
				{ 1.0, 0.0, 1.0 },
				{ 0.0, 1.0, 0.0 }
			}
		}
	));
	i_data[1] = unique_ptr<::data<FLT>>(new tns<FLT>(
		{
			{
				{ 1.0, 0.0, 1.0 },
				{ 0.0, 0.0, 0.0 },
				{ 1.0, 0.0, 1.0 }
			}
		}
	));

	o_data[0] = unique_ptr<::data<FLT>>(new tns<FLT>(
		{
			{
				{ 1.0 },
				{ 0.0 }
			}
		}
	));
	o_data[1] = unique_ptr<::data<FLT>>(new tns<FLT>(
		{
			{
				{ 0.0 },
				{ 1.0 }
			}
		}
	));

	pipe<info> pip;
	wx_wrapper wx;
	wx.create_graph_wnd(pip);

	nnb network(
		{
			unique_ptr<nn_info>((nn_info*)new cnn_info
				(2, 1, 3, 3,nn_params::nn_activ_t::relu, nn_params::nn_init_t::normal, 1, 1, 1, false))
		}
	);

	network.train({ move(i_data), move(o_data), 2 }, { nullptr,nullptr,0 }, train_mode::NONE, 0, &pip, 0, 10000, 0, 0, 1, 0.95, 0.95, 0.0095, 0.00095);*/

//tns<FLT> input_tensor = {
//{
//	{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
//	{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
//	{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
//	{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
//	{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
//	{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
//	{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
//	{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
//	{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
//},
//{
//	{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
//	{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
//	{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
//	{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
//	{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
//	{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
//	{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
//	{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
//	{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}
//}
//};
//
//tns<FLT> core_tensor = {
//		{
//			{1.0, 0.0, -1.0},
//			{1.0, 0.0, -1.0},
//			{1.0, 0.0, -1.0}
//		},
//		{
//			{1.0, 1.0, 1.0},
//			{0.0, 0.0, 0.0},
//			{-1.0, -1.0, -1.0}
//		},
//		{
//			{0.0, 1.0, 0.0},
//			{1.0, -4.0, 1.0},
//			{0.0, 1.0, 0.0}
//		},
//		{
//			{1.0, 2.0, 1.0},
//			{0.0, 0.0, 0.0},
//			{-1.0, -2.0, -1.0}
//		},
//		{
//			{1.0, 0.0, -1.0},
//			{2.0, 0.0, -2.0},
//			{1.0, 0.0, -1.0}
//		},
//		{
//			{0.0, 1.0, 0.0},
//			{1.0, -4.0, 1.0},
//			{0.0, 1.0, 0.0}
//		}
//};
//
//tns<FLT> gradient_tensor = {
//	{
//		{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
//		{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
//		{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
//		{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
//		{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
//		{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
//		{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
//		{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
//		{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
//	},
//	{
//		{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 },
//		{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 },
//		{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 },
//		{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 },
//		{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 },
//		{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 },
//		{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 },
//		{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 },
//		{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 }
//	},
//	{
//		{ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 },
//		{ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 },
//		{ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 },
//		{ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 },
//		{ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 },
//		{ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 },
//		{ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 },
//		{ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 },
//		{ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 }
//	}
//};
//
//vec<bool> drop({ true, true, true });
//vec<bool> drop_2({ true, true });
//
//tns<FLT> res = arithmetic::convolute(input_tensor, core_tensor);
//tns<FLT> res_2(3, 9, 9);
//arithmetic::convolute_fwd(input_tensor, core_tensor, drop, res_2);
//
//tns<FLT> grad_input(2, 9, 9);
//tns<FLT> delta_core(6, 3, 3);
//
//arithmetic::convolute_bwd(gradient_tensor, core_tensor, drop_2, grad_input);
//arithmetic::convolute_rev(input_tensor, gradient_tensor, drop, delta_core);
//
//delta_core *= (FLT)81;

	getchar();
}