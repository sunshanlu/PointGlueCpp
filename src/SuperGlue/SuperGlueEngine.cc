#include "pointgluecpp/SuperGlue/SuperGlueEngine.h"

#define RET_OK nullptr
#define CHECK(message)\
do {\
	if (message != RET_OK){\
		SPDLOG_ERROR("Error: {}, at {}:{}\n", message, __FILE__, __LINE__);\
		exit(EXIT_FAILURE);\
	}\
} while (0)

namespace sg {
SuperGlueEngine::SuperGlueEngine(const std::string &model_path,
                                 const std::vector<const char *> &input_names,
                                 const std::vector<const char *> &output_names,
                                 const short &desc_dim,
                                 const short &keypoint_dim)
	: model_path_(model_path),
	  input_names_(input_names),
	  output_names_(output_names),
	  desc_dim_(desc_dim),
	  keypoint_dim_(keypoint_dim) {
}
} // namespace sg

#undef RET_OK
#undef CHECK