#include <cstdint>
#include <ctime>
#include <random>

#include <CppUnitTest.h>

#include "../NeuralNetwork/rfm.h"
#include "../NeuralNetwork/cnn.h"
#include "../NeuralNetwork/fnn.h"
#include "../NeuralNetwork/rnn.h"
#include "../NeuralNetwork/nnb.h"

#include "../NeuralNetwork/fnn_params.h"
#include "../NeuralNetwork/fnn_activs.h"
#include "../NeuralNetwork/fnn_derivs.h"
#include "../NeuralNetwork/fnn_inits.h"

using namespace std;
using namespace fnn_params;
using namespace fnn_activs;
using namespace fnn_derivs;
using namespace arithmetic;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTests
{
	TEST_CLASS(UnitTests)
	{
	public:
		TEST_METHOD(FNNActivationTest)
		{
			Assert::AreEqual((double)1, activation<fnn_activ::signed_pos>((double)2,(double)1));
			Assert::AreEqual((double)1, activation<fnn_activ::signed_pos>((double)0, (double)1));
			Assert::AreEqual((double)-1, activation<fnn_activ::signed_pos>((double)-2, (double)1));

			Assert::AreEqual((double)1, activation<fnn_activ::signed_neg>((double)2, (double)1));
			Assert::AreEqual((double)-1, activation<fnn_activ::signed_neg>((double)0, (double)1));
			Assert::AreEqual((double)-1, activation<fnn_activ::signed_neg>((double)-2, (double)1));

			Assert::AreEqual((double)1, activation<fnn_activ::thresh_pos>((double)2, (double)1));
			Assert::AreEqual((double)1, activation<fnn_activ::thresh_pos>((double)0, (double)1));
			Assert::AreEqual((double)0, activation<fnn_activ::thresh_pos>((double)-2, (double)1));

			Assert::AreEqual((double)1, activation<fnn_activ::thresh_neg>((double)2, (double)1));
			Assert::AreEqual((double)0, activation<fnn_activ::thresh_neg>((double)0, (double)1));
			Assert::AreEqual((double)0, activation<fnn_activ::thresh_neg>((double)-2, (double)1));

			Assert::AreEqual((double)1, activation<fnn_activ::step_sym>((double)2, (double)1));
			Assert::AreEqual((double)0.5, activation<fnn_activ::step_sym>((double)0.5, (double)1));
			Assert::AreEqual((double)-1, activation<fnn_activ::step_sym>((double)-2, (double)1));

			Assert::AreEqual((double)1, activation<fnn_activ::step_pos>((double)2, (double)1));
			Assert::AreEqual((double)0.5, activation<fnn_activ::step_pos>((double)0.5, (double)1));
			Assert::AreEqual((double)0, activation<fnn_activ::step_pos>((double)-2, (double)1));

			Assert::AreEqual((double)0, activation<fnn_activ::step_neg>((double)2, (double)1));
			Assert::AreEqual((double)-0.5, activation<fnn_activ::step_neg>((double)-0.5, (double)1));
			Assert::AreEqual((double)-1, activation<fnn_activ::step_neg>((double)-2, (double)1));

			Assert::AreEqual((double)1, activation<fnn_activ::rad_bas_pos>((double)0, (double)1));

			Assert::AreEqual((double)-1, activation<fnn_activ::rad_bas_neg>((double)0, (double)1));

			Assert::AreEqual((double)0.5, activation<fnn_activ::sigmoid_log>((double)0, (double)1));

			Assert::AreEqual((double)0, activation<fnn_activ::sigmoid_rat>((double)0, (double)1));

			Assert::AreEqual((double)0, activation<fnn_activ::atan>((double)0, (double)1));

			Assert::AreEqual((double)0, activation<fnn_activ::tanh>((double)0, (double)1));

			Assert::AreEqual((double)0, activation<fnn_activ::elu>((double)0, (double)1));

			Assert::AreEqual((double)0, activation<fnn_activ::gelu>((double)0, (double)1));

			Assert::AreEqual((double)1, activation<fnn_activ::lelu>((double)1, (double)0.0625));
			Assert::AreEqual((double)0, activation<fnn_activ::lelu>((double)0, (double)0.0625));
			Assert::AreEqual((double)-0.0625, activation<fnn_activ::lelu>((double)-1, (double)0.0625));

			Assert::AreEqual((double)1, activation<fnn_activ::relu>((double)1, (double)1));
			Assert::AreEqual((double)0, activation<fnn_activ::relu>((double)0, (double)1));
			Assert::AreEqual((double)0, activation<fnn_activ::relu>((double)-1, (double)1));

			Assert::AreEqual((double)0, activation<fnn_activ::mish>((double)0, (double)1));

			Assert::AreEqual((double)0, activation<fnn_activ::swish>((double)0, (double)1));

			Assert::AreEqual((double)0, activation<fnn_activ::softplus>((double)-100, (double)1), (double)0.001);
		}

		TEST_METHOD(FNNDerivationTest)
		{
			// FUCKEN HARD
		}

		TEST_METHOD(FNNInitiationTest)
		{
			// FUCKEN HARD
		}

		TEST_METHOD(ArithmeticTest)
		{
			const uint64_t len_1 = (uint64_t)5, len_2 = (uint64_t)3;

			int vec_data[len_1] = { 4, 2, 0, 1, 3 };
			int res_1_data[len_2] = { 42, 38, 71 };
			int res_2_data[len_2] = { 23, 39, 58 };

			int mtx_1_data[len_1][len_2] =
			{
				{ 1, 3, 9 },
				{ 6, 5, 7 },
				{ 6, 2, 8 },
				{ 2, 1, 9 },
				{ 8, 5, 4 }
			};

			int mtx_2_data[len_2][len_1] =
			{
				{ 0, 4, 5, 9, 2 },
				{ 3, 4, 7, 1, 6 },
				{ 5, 7, 7, 0, 8 }
			};

			vec<int> vec_1(len_1);
			mtx<int> mtx_1(len_1, len_2);
			mtx<int> mtx_2(len_2, len_1);

			for (uint64_t i = (uint64_t)0; i < len_1; ++i)
				vec_1(i) = vec_data[i];

			for (uint64_t i = (uint64_t)0; i < len_1; ++i)
				for (uint64_t j = (uint64_t)0; j < len_2; ++j)
					mtx_1(i, j) = mtx_1_data[i][j];

			for (uint64_t i = (uint64_t)0; i < len_2; ++i)
				for (uint64_t j = (uint64_t)0; j < len_1; ++j)
					mtx_2(i, j) = mtx_2_data[i][j];

			vec<int> vec_2 = vec_1 * mtx_1;
			vec<int> vec_3 = mtx_2 * vec_1;

			for (uint64_t i = (uint64_t)0; i < len_2; ++i)
			{
				Assert::AreEqual(res_1_data[i], vec_2(i));
				Assert::AreEqual(res_2_data[i], vec_3(i));
			}
		}

		TEST_METHOD(MatrixTest)
		{
			constexpr uint64_t size = 100;
			uint64_t seed = (uint64_t)time(nullptr);
			mtx<double> mtx_1(size, size);
			void* ptr = (void*)(&mtx_1((uint64_t)0, (uint64_t)0));

			srand(seed);

			for (uint64_t i = (uint64_t)0; i < size; ++i)
				for (uint64_t j = (uint64_t)0; j < size; ++j)
					mtx_1(i, j) = (double)rand();

			mtx<double> mtx_2(move(mtx_1));
			mtx<double> mtx_3(mtx_2);

			Assert::AreEqual(true, mtx_1.is_empty());
			Assert::AreEqual((uint64_t)0, mtx_1.get_size_1());
			Assert::AreEqual((uint64_t)0, mtx_1.get_size_2());

			Assert::AreEqual(false, mtx_2.is_empty());
			Assert::AreEqual(size, mtx_2.get_size_1());
			Assert::AreEqual(size, mtx_2.get_size_2());

			Assert::AreEqual(false, mtx_3.is_empty());
			Assert::AreEqual(size, mtx_3.get_size_1());
			Assert::AreEqual(size, mtx_3.get_size_2());

			Assert::AreEqual(ptr, (void*)(&mtx_2((uint64_t)0, (uint64_t)0)));
			Assert::AreNotEqual(ptr, (void*)(&mtx_3((uint64_t)0, (uint64_t)0)));

			srand(seed);

			for (uint64_t i = (uint64_t)0; i < size; ++i)
				for (uint64_t j = (uint64_t)0; j < size; ++j)
				{
					double data = (double)rand();

					Assert::AreEqual(data, mtx_2(i, j));
					Assert::AreEqual(data, mtx_3(i, j));
				}

			mtx<double> mtx_4;
			mtx<double> mtx_5;

			Assert::AreEqual(true, mtx_4.is_empty());
			Assert::AreEqual((uint64_t)0, mtx_4.get_size_1());
			Assert::AreEqual((uint64_t)0, mtx_4.get_size_2());

			Assert::AreEqual(true, mtx_5.is_empty());
			Assert::AreEqual((uint64_t)0, mtx_5.get_size_1());
			Assert::AreEqual((uint64_t)0, mtx_5.get_size_2());

			mtx_4 = move(mtx_2);
			mtx_5 = mtx_3;

			Assert::AreEqual(true, mtx_2.is_empty());
			Assert::AreEqual((uint64_t)0, mtx_2.get_size_1());
			Assert::AreEqual((uint64_t)0, mtx_2.get_size_2());

			Assert::AreEqual(false, mtx_3.is_empty());
			Assert::AreEqual(size, mtx_3.get_size_1());
			Assert::AreEqual(size, mtx_3.get_size_2());

			Assert::AreEqual(false, mtx_4.is_empty());
			Assert::AreEqual(size, mtx_4.get_size_1());
			Assert::AreEqual(size, mtx_4.get_size_2());

			Assert::AreEqual(false, mtx_5.is_empty());
			Assert::AreEqual(size, mtx_5.get_size_1());
			Assert::AreEqual(size, mtx_5.get_size_2());

			Assert::AreEqual(ptr, (void*)(&mtx_4((uint64_t)0, (uint64_t)0)));
			Assert::AreNotEqual(ptr, (void*)(&mtx_3((uint64_t)0, (uint64_t)0)));
			Assert::AreNotEqual(ptr, (void*)(&mtx_5((uint64_t)0, (uint64_t)0)));
			Assert::AreNotEqual((void*)(&mtx_3((uint64_t)0, (uint64_t)0)), (void*)(&mtx_5((uint64_t)0, (uint64_t)0)));

			srand(seed);

			for (uint64_t i = (uint64_t)0; i < size; ++i)
				for (uint64_t j = (uint64_t)0; j < size; ++j)
				{
					double data = (double)rand();

					Assert::AreEqual(data, mtx_4(i, j));
					Assert::AreEqual(data, mtx_5(i, j));
				}
		}

		TEST_METHOD(VectorTest)
		{
			constexpr uint64_t size = 100;
			uint64_t seed = (uint64_t)time(nullptr);
			vec<double> vec_1(size);
			void* ptr = (void*)(&vec_1((uint64_t)0));

			srand(seed);

			for (uint64_t i = (uint64_t)0; i < size; ++i)
				vec_1(i) = (double)rand();

			vec<double> vec_2(move(vec_1));
			vec<double> vec_3(vec_2);

			Assert::AreEqual(true, vec_1.is_empty());
			Assert::AreEqual((uint64_t)0, vec_1.get_size());

			Assert::AreEqual(false, vec_2.is_empty());
			Assert::AreEqual(size, vec_2.get_size());

			Assert::AreEqual(false, vec_3.is_empty());
			Assert::AreEqual(size, vec_3.get_size());

			Assert::AreEqual(ptr, (void*)(&vec_2((uint64_t)0)));
			Assert::AreNotEqual(ptr, (void*)(&vec_3((uint64_t)0)));

			srand(seed);

			for (uint64_t i = (uint64_t)0; i < size; ++i)
			{
				double data = (double)rand();

				Assert::AreEqual(data, vec_2(i));
				Assert::AreEqual(data, vec_3(i));
			}

			vec<double> vec_4;
			vec<double> vec_5;

			Assert::AreEqual(true, vec_4.is_empty());
			Assert::AreEqual((uint64_t)0, vec_4.get_size());

			Assert::AreEqual(true, vec_5.is_empty());
			Assert::AreEqual((uint64_t)0, vec_5.get_size());

			vec_4 = move(vec_2);
			vec_5 = vec_3;

			Assert::AreEqual(true, vec_2.is_empty());
			Assert::AreEqual((uint64_t)0, vec_2.get_size());

			Assert::AreEqual(false, vec_3.is_empty());
			Assert::AreEqual(size, vec_3.get_size());

			Assert::AreEqual(false, vec_4.is_empty());
			Assert::AreEqual(size, vec_4.get_size());

			Assert::AreEqual(false, vec_5.is_empty());
			Assert::AreEqual(size, vec_5.get_size());

			Assert::AreEqual(ptr, (void*)(&vec_4((uint64_t)0)));
			Assert::AreNotEqual(ptr, (void*)(&vec_3((uint64_t)0)));
			Assert::AreNotEqual(ptr, (void*)(&vec_5((uint64_t)0)));
			Assert::AreNotEqual((void*)(&vec_3((uint64_t)0)), (void*)(&vec_5((uint64_t)0)));

			srand(seed);

			for (uint64_t i = (uint64_t)0; i < size; ++i)
			{
				double data = (double)rand();

				Assert::AreEqual(data, vec_4(i));
				Assert::AreEqual(data, vec_5(i));
			}
		}
	};
}