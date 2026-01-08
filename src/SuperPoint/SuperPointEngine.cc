#include "pointgluecpp/SuperPoint/SuperPointEngine.h"


#define RET_OK nullptr
#define CHECK(message)\
do {\
	if (message != RET_OK){\
		SPDLOG_ERROR("Error: {}, at {}:{}\n", message, __FILE__, __LINE__);\
		exit(EXIT_FAILURE);\
	}\
} while (0)

namespace sp {
SuperPointEngine::SuperPointEngine(
	const std::string &model_path,
	std::vector<const char *> input_names,
	std::vector<const char *> output_names,
	std::vector<int64_t> input_dims,
	std::vector<int64_t> output_dims0,
	std::vector<int64_t> output_dims1
) : model_path_(model_path)
    , input_names_(std::move(input_names))
    , output_names_(std::move(output_names))
    , input_dims_(std::move(input_dims))
    , output_dims0_(std::move(output_dims0))
    , output_dims1_(std::move(output_dims1)) {
}
}
