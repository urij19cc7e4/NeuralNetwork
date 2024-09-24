#include "nn.h"
#include "cnn.h"
#include "fnn.h"
#include "cnn_2_fnn.h"

nn* nn::create_from_file(std::ifstream& file)
{
	uint64_t id;
	file.read(reinterpret_cast<char*>(&id), sizeof(id));

	switch (id)
	{
	case (uint64_t)layer_id::CNN:
		return (nn*)(new cnn(file));

	case (uint64_t)layer_id::FNN:
		return (nn*)(new fnn(file));

	case (uint64_t)adapter_id::CNN_2_FNN:
		return (nn*)(new cnn_2_fnn(file));

	default:
		throw std::exception("...");
	}
}