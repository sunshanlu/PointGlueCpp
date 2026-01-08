#pragma once

#include <fstream>
#include <string>

#include <spdlog/spdlog.h>

#define RET_OK nullptr
#define CHECK(message)                                                                                                                               \
  do {                                                                                                                                               \
    if (message != RET_OK) {                                                                                                                         \
      SPDLOG_ERROR("Error: {}, at {}:{}\n", message, __FILE__, __LINE__);                                                                            \
      exit(EXIT_FAILURE);                                                                                                                            \
    }                                                                                                                                                \
  } while (0)

template <typename T>
const char *SaveDataToFile(
  const T *tensor_data, 
  const std::size_t size, 
  const std::string &file_path) {
  std::ofstream outfile(file_path);

  if (!outfile.is_open())
    return "Error: can not open file";

  for (size_t idx = 0; idx < size; ++idx)
    outfile << tensor_data[idx] << " ";

  outfile << std::endl;

  return RET_OK;
}

#undef RET_OK
#undef CHECK
