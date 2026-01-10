#pragma once

#include <fstream>
#include <string>

#include <spdlog/spdlog.h>

/**
 * @file DebugTools.hpp
 * @brief 调试工具函数和宏定义
 *
 * 提供张量数据保存和错误检查等调试辅助功能
 */

/**
 * @brief 成功返回值宏定义
 */
#define RET_OK nullptr

/**
 * @brief 错误检查宏
 *
 * 检查返回值，如果不为 RET_OK 则记录错误日志并退出程序
 *
 * @param message 待检查的返回值
 */
#define CHECK(message)                                                                                                                               \
  do {                                                                                                                                               \
    if (message != RET_OK) {                                                                                                                         \
      SPDLOG_ERROR("Error: {}, at {}:{}\n", message, __FILE__, __LINE__);                                                                            \
      exit(EXIT_FAILURE);                                                                                                                            \
    }                                                                                                                                                \
  } while (0)

/**
 * @brief 将张量数据保存到文件
 *
 * 将数组中的数据以空格分隔的形式写入文本文件
 *
 * @tparam T 数据类型（支持流输出操作符的类型）
 * @param tensor_data 张量数据指针
 * @param size 数据元素数量
 * @param file_path 输出文件路径
 * @return const char* 错误信息，成功返回 nullptr
 */
template <typename T>
const char* SaveDataToFile(
  const T* tensor_data,
  const std::size_t size,
  const std::string& file_path)
{
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
