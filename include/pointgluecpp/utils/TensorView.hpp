#pragma once

#include <numeric>
#include <vector>

/**
 * @file TensorView.hpp
 * @brief 零拷贝张量视图模板类
 *
 * 提供高效的张量数据包装器，支持拥有和非拥有两种模式
 * 支持零拷贝视图和深拷贝语义
 */

/**
 * @brief 张量视图模板类
 *
 * @tparam T 元素类型（如 float, int32_t 等）
 *
 * 提供对连续数组的轻量级包装，支持：
 * - 拥有模式：管理内存生命周期（深拷贝）
 * - 非拥有模式：仅包装外部指针（零拷贝视图）
 * - 拷贝和移动语义
 *
 * @note 当 size=0 时，不分配内存，创建空视图
 */
template <class T>
struct TensorView
{
public:
  /**
   * @brief 默认构造函数
   *
   * 创建空的 TensorView，不分配内存
   */
  TensorView()
    : data_(nullptr),
      size_(0),
      owner_ship_(false) {}

  /**
   * @brief 指定大小的构造函数
   *
   * @param size 张量元素数量
   *
   * @note 当 size=0 时，不分配内存，创建空 TensorView
   * @note 当 size>0 时，分配内存并拥有所有权
   */
  explicit TensorView(const std::size_t& size)
    : data_(nullptr),
      size_(size),
      owner_ship_(size > 0) { if (size > 0) { data_ = new T[size]; } }

  /**
   * @brief 从外部指针创建视图（非拥有模式）
   *
   * @param data 外部数据指针
   * @param size 数据元素数量
   *
   * @note 不获取内存所有权，析构时不会释放内存
   */
  TensorView(T* data, const std::size_t& size)
    : data_(data),
      size_(size),
      owner_ship_(false) {}

  /**
   * @brief 从外部指针和维度数组创建视图
   *
   * @param data 外部数据指针
   * @param sizes 各维度的大小列表
   *
   * @note 自动计算总元素数量（各维度乘积）
   */
  template <class S>
  TensorView(T* data, const std::vector<S>& sizes)
    : data_(data),
      owner_ship_(false) { size_ = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<S>()); }

  /**
   * @brief 拷贝构造函数
   *
   * @param rhs 源对象
   *
   * 如果源对象拥有内存所有权，则执行深拷贝
   * 否则仅复制指针（浅拷贝）
   */
  TensorView(const TensorView& rhs)
    : data_(nullptr),
      size_(rhs.size_),
      owner_ship_(rhs.owner_ship_)
  {
    if (owner_ship_ && size_ > 0)
    {
      data_ = new T[rhs.size_];
      memcpy(data_, rhs.data_, rhs.size_ * sizeof(T));
    }
    else { data_ = rhs.data_; }
  }

  /**
   * @brief 移动构造函数
   *
   * @param rhs 源对象
   *
   * 转移内存所有权，源对象变为空
   */
  TensorView(TensorView&& rhs) noexcept
    : data_(rhs.data_),
      size_(rhs.size_),
      owner_ship_(rhs.owner_ship_)
  {
    rhs.owner_ship_ = false;
    rhs.data_ = nullptr;
    rhs.size_ = 0;
  }

  /**
   * @brief 拷贝赋值运算符
   *
   * @param rhs 源对象
   * @return TensorView& *this
   *
   * 正确处理内存所有权：
   * - 如果源对象拥有所有权，执行深拷贝
   * - 否则执行浅拷贝（仅复制指针）
   */
  TensorView& operator=(const TensorView& rhs)
  {
    if (this != &rhs)
    {
      // 释放当前对象的内存（如果拥有所有权）
      if (owner_ship_)
      {
        delete[] data_;
        data_ = nullptr;
      }

      // 复制数据
      size_ = rhs.size_;
      owner_ship_ = rhs.owner_ship_;

      if (owner_ship_ && size_ > 0)
      {
        // 深拷贝：分配新内存并复制数据
        data_ = new T[rhs.size_];
        memcpy(data_, rhs.data_, rhs.size_ * sizeof(T));
      }
      else
      {
        // 浅拷贝：只复制指针，不拥有所有权（或 size 为 0）
        data_ = rhs.data_;
      }
    }
    return *this;
  }

  /**
   * @brief 移动赋值运算符
   *
   * @param rhs 源对象
   * @return TensorView& *this
   *
   * 转移内存所有权，释放旧内存，源对象变为空
   */
  TensorView& operator=(TensorView&& rhs) noexcept
  {
    if (this != &rhs)
    {
      // 释放当前对象的内存（如果拥有所有权）
      if (owner_ship_) { delete[] data_; }

      // 转移所有权和数据
      data_ = rhs.data_;
      size_ = rhs.size_;
      owner_ship_ = rhs.owner_ship_;

      // 将源对象重置为安全状态
      rhs.data_ = nullptr;
      rhs.size_ = 0;
      rhs.owner_ship_ = false;
    }
    return *this;
  }

  /**
   * @brief 获取数据指针
   *
   * @return T* 数据指针
   */
  T* data() { return data_; }

  /**
   * @brief 获取数据指针（const 版本）
   *
   * @return const T* 数据指针
   */
  const T* data() const { return data_; }

  /**
   * @brief 析构函数
   *
   * 仅在拥有内存所有权时释放内存
   */
  ~TensorView()
  {
    if (owner_ship_)
      delete[] data_;
  }

  T* data_ = nullptr;       ///< 数据指针
  std::size_t size_ = 0;    ///< 数据元素数量
  bool owner_ship_ = false; ///< 是否拥有内存所有权
};
