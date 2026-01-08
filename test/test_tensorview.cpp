#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "pointgluecpp/utils/TensorView.hpp"

/**
 * TensorView 单元测试
 *
 * 测试重点：
 * 1. 内存所有权管理
 * 2. 拷贝/移动语义
 * 3. 数据正确性
 * 4. 边界条件
 */
class TensorViewTest : public ::testing::Test
{
protected:
  void SetUp() override {}

  void TearDown() override {}
};

// =============================================================================
// 测试 1: 基本构造和内存所有权
// =============================================================================

TEST_F(TensorViewTest, ConstructWithOwnerShip)
{
  // 测试拥有所有权的构造
  constexpr std::size_t size = 100;
  TensorView<float> tv(size);

  EXPECT_NE(tv.data(), nullptr);
  EXPECT_EQ(tv.size_, size);
  EXPECT_TRUE(tv.owner_ship_);

  // 填充数据
  std::vector<float> expected(size);
  for (std::size_t i = 0; i < size; ++i)
  {
    expected[i] = static_cast<float>(i);
    tv.data()[i] = expected[i];
  }

  // 验证数据
  for (std::size_t i = 0; i < size; ++i) { EXPECT_FLOAT_EQ(tv.data()[i], expected[i]); }
}

TEST_F(TensorViewTest, ConstructWithoutOwnerShip)
{
  // 测试不拥有所有权的构造
  constexpr std::size_t size = 100;
  std::vector<float> data(size);
  for (std::size_t i = 0; i < size; ++i) { data[i] = static_cast<float>(i); }

  {
    TensorView tv(data.data(), size);
    EXPECT_EQ(tv.data(), data.data());
    EXPECT_EQ(tv.size_, size);
    EXPECT_FALSE(tv.owner_ship_);
  }
  // data 在这里仍然有效，验证悬空指针问题
  EXPECT_EQ(data.size(), size);
}

TEST_F(TensorViewTest, ConstructWithVectorSizes)
{
  // 测试使用维度向量构造
  constexpr std::size_t size = 100;
  std::vector<float> data(size);
  const std::vector<std::size_t> sizes = {10, 10};

  TensorView tv(data.data(), sizes);
  EXPECT_EQ(tv.data(), data.data());
  EXPECT_EQ(tv.size_, size);
  EXPECT_FALSE(tv.owner_ship_);
}

// =============================================================================
// 测试 2: 拷贝构造 - 发现内存管理问题
// =============================================================================

TEST_F(TensorViewTest, CopyConstructorWithOwnership)
{
  // 测试拥有所有权对象的拷贝构造
  constexpr std::size_t size = 100;
  TensorView<float> original(size);

  // 填充数据
  for (std::size_t i = 0; i < size; ++i) { original.data()[i] = static_cast<float>(i); }

  TensorView copy(original);

  // 验证数据独立性
  EXPECT_NE(copy.data(), original.data());
  EXPECT_EQ(copy.size_, original.size_);
  EXPECT_TRUE(copy.owner_ship_);
  EXPECT_TRUE(original.owner_ship_);

  // 验证数据内容
  for (std::size_t i = 0; i < size; ++i) { EXPECT_FLOAT_EQ(copy.data()[i], original.data()[i]); }

  // 修改 copy 不应影响 original
  copy.data()[0] = 999.0f;
  EXPECT_NE(copy.data()[0], original.data()[0]);
}

TEST_F(TensorViewTest, CopyConstructorWithoutOwnership)
{
  // 测试不拥有所有权对象的拷贝构造 - 这是一个风险点！
  std::vector data(100, 42.0f);
  TensorView original(data.data(), 100);

  TensorView copy(original);

  // 🔴 风险点：浅拷贝可能导致双重释放或悬空指针
  EXPECT_EQ(copy.data(), original.data());
  EXPECT_EQ(copy.size_, original.size_);
  EXPECT_FALSE(copy.owner_ship_);
  EXPECT_FALSE(original.owner_ship_);

  // 两个对象指向同一块内存
  EXPECT_EQ(copy.data()[0], original.data()[0]);
}

// =============================================================================
// 测试 3: 移动构造
// =============================================================================

TEST_F(TensorViewTest, MoveConstructorWithOwnership)
{
  constexpr std::size_t size = 100;
  TensorView<float> original(size);
  const float* original_ptr = original.data();

  // 填充数据
  for (std::size_t i = 0; i < size; ++i) { original.data()[i] = static_cast<float>(i); }

  TensorView moved(std::move(original));

  // 验证所有权转移
  EXPECT_EQ(moved.data(), original_ptr);
  EXPECT_EQ(moved.size_, size);
  EXPECT_TRUE(moved.owner_ship_);

  // original 应该被清空
  EXPECT_EQ(original.data_, nullptr);
  EXPECT_EQ(original.size_, 0);
  EXPECT_FALSE(original.owner_ship_);

  // 验证数据完整性
  for (std::size_t i = 0; i < size; ++i) { EXPECT_FLOAT_EQ(moved.data()[i], static_cast<float>(i)); }
}

TEST_F(TensorViewTest, MoveConstructorWithoutOwnership)
{
  std::vector data(100, 42.0f);
  TensorView original(data.data(), 100);

  TensorView moved(std::move(original));

  // 验证指针转移
  EXPECT_EQ(moved.data(), data.data());
  EXPECT_EQ(moved.size_, 100);
  EXPECT_FALSE(moved.owner_ship_);

  // original 应该被清空
  EXPECT_EQ(original.data_, nullptr);
  EXPECT_EQ(original.size_, 0);
  EXPECT_FALSE(original.owner_ship_);
}

// =============================================================================
// 测试 4: 析构和内存泄漏
// =============================================================================

TEST_F(TensorViewTest, DestructorWithOwnership)
{
  // 测试析构函数是否正确释放内存
  constexpr std::size_t size = 100;
  auto* tv = new TensorView<float>(size);
  float* data_ptr = tv->data();

  EXPECT_NE(data_ptr, nullptr);

  delete tv;
  // 如果没有崩溃，说明析构正常工作
  // 注意：这里无法验证内存是否真的被释放，需要使用 Valgrind 等工具
}

TEST_F(TensorViewTest, DestructorWithoutOwnership)
{
  // 测试不拥有所有权时的析构
  std::vector<float> data(100, 42.0f);
  const auto* tv = new TensorView(data.data(), 100);

  delete tv;
  // data 在这里仍然应该有效
  EXPECT_EQ(data[0], 42.0f);
}

// =============================================================================
// 测试 5: 拷贝赋值运算符
// =============================================================================

TEST_F(TensorViewTest, CopyAssignmentOperatorWithOwnership)
{
  // 测试拷贝赋值运算符 - 拥有所有权
  constexpr std::size_t size = 100;
  TensorView<float> original(size);
  TensorView<float> another(50); // 不同的初始大小

  // 填充 original
  for (std::size_t i = 0; i < size; ++i) { original.data()[i] = static_cast<float>(i); }

  // 执行拷贝赋值
  another = original;

  // 验证赋值结果
  EXPECT_NE(another.data(), original.data());
  EXPECT_EQ(another.size_, original.size_);
  EXPECT_TRUE(another.owner_ship_);
  EXPECT_TRUE(original.owner_ship_);

  // 验证数据内容
  for (std::size_t i = 0; i < size; ++i) { EXPECT_FLOAT_EQ(another.data()[i], original.data()[i]); }

  // 修改 another 不应影响 original
  another.data()[0] = 999.0f;
  EXPECT_NE(another.data()[0], original.data()[0]);
}

TEST_F(TensorViewTest, CopyAssignmentOperatorWithoutOwnership)
{
  // 测试拷贝赋值运算符 - 不拥有所有权
  std::vector data1(100, 42.0f);
  std::vector data2(50, 24.0f);
  TensorView original(data1.data(), 100);
  TensorView another(data2.data(), 50);

  // 执行拷贝赋值
  another = original;

  // 验证赋值结果 - 浅拷贝
  EXPECT_EQ(another.data(), original.data());
  EXPECT_EQ(another.size_, 100);
  EXPECT_FALSE(another.owner_ship_);
  EXPECT_FALSE(original.owner_ship_);
}

TEST_F(TensorViewTest, CopyAssignmentSelfAssignment)
{
  // 测试自赋值
  constexpr std::size_t size = 100;
  TensorView<float> tv(size);
  const float* ptr = tv.data();

  // 填充数据
  for (std::size_t i = 0; i < size; ++i) { tv.data()[i] = static_cast<float>(i); }

  // 自赋值
  tv = tv;

  // 验证指针未变，数据未变
  EXPECT_EQ(tv.data(), ptr);
  EXPECT_EQ(tv.size_, size);
  for (std::size_t i = 0; i < size; ++i) { EXPECT_FLOAT_EQ(tv.data()[i], static_cast<float>(i)); }
}

TEST_F(TensorViewTest, CopyAssignmentOwnershipTransfer)
{
  // 测试从拥有所有权赋值给不拥有所有权
  constexpr std::size_t size = 100;
  TensorView<float> original(size);
  std::vector data(50, 24.0f);
  TensorView another(data.data(), 50);

  // 填充 original
  for (std::size_t i = 0; i < size; ++i) { original.data()[i] = static_cast<float>(i); }

  EXPECT_TRUE(original.owner_ship_);
  EXPECT_FALSE(another.owner_ship_);

  // 执行拷贝赋值
  another = original;

  // 验证：another 应该拥有所有权并深拷贝数据
  EXPECT_NE(another.data(), original.data());
  EXPECT_EQ(another.size_, size);
  EXPECT_TRUE(another.owner_ship_);
  EXPECT_TRUE(original.owner_ship_);

  // data 应该仍然有效
  EXPECT_EQ(data[0], 24.0f);
}

// =============================================================================
// 测试 6: 移动赋值运算符
// =============================================================================

TEST_F(TensorViewTest, MoveAssignmentOperatorWithOwnership)
{
  // 测试移动赋值运算符 - 拥有所有权
  constexpr std::size_t size = 100;
  TensorView<float> original(size);
  TensorView<float> another(50); // 不同的初始大小

  const float* original_ptr = original.data();

  // 填充 original
  for (std::size_t i = 0; i < size; ++i) { original.data()[i] = static_cast<float>(i); }

  // 执行移动赋值
  another = std::move(original);

  // 验证所有权转移
  EXPECT_EQ(another.data(), original_ptr);
  EXPECT_EQ(another.size_, size);
  EXPECT_TRUE(another.owner_ship_);

  // original 应该被清空
  EXPECT_EQ(original.data_, nullptr);
  EXPECT_EQ(original.size_, 0);
  EXPECT_FALSE(original.owner_ship_);

  // 验证数据完整性
  for (std::size_t i = 0; i < size; ++i)
    EXPECT_FLOAT_EQ(another.data()[i], static_cast<float>(i));
}

TEST_F(TensorViewTest, MoveAssignmentOperatorWithoutOwnership)
{
  // 测试移动赋值运算符 - 不拥有所有权
  std::vector data(100, 42.0f);
  TensorView original(data.data(), 100);
  TensorView<float> another(50);

  // 执行移动赋值
  another = std::move(original);

  // 验证指针转移
  EXPECT_EQ(another.data(), data.data());
  EXPECT_EQ(another.size_, 100);
  EXPECT_FALSE(another.owner_ship_);

  // original 应该被清空
  EXPECT_EQ(original.data_, nullptr);
  EXPECT_EQ(original.size_, 0);
  EXPECT_FALSE(original.owner_ship_);
}

TEST_F(TensorViewTest, MoveAssignmentSelfAssignment)
{
  // 测试自移动赋值
  constexpr std::size_t size = 100;
  TensorView<float> tv(size);
  const float* ptr = tv.data();

  // 填充数据
  for (std::size_t i = 0; i < size; ++i)
    tv.data()[i] = static_cast<float>(i);

  // 自移动赋值
  tv = std::move(tv);

  // 验证指针未变，数据未变
  EXPECT_EQ(tv.data(), ptr);
  EXPECT_EQ(tv.size_, size);
  for (std::size_t i = 0; i < size; ++i)
    EXPECT_FLOAT_EQ(tv.data()[i], static_cast<float>(i));
}

TEST_F(TensorViewTest, MoveAssignmentOwnershipTransfer)
{
  // 测试移动赋值时的所有权转移
  constexpr std::size_t size = 100;
  TensorView<float> original(size);
  TensorView<float> another(50);

  EXPECT_TRUE(original.owner_ship_);
  EXPECT_TRUE(another.owner_ship_);

  // 执行移动赋值
  another = std::move(original);

  // 验证所有权转移
  EXPECT_TRUE(another.owner_ship_);
  EXPECT_FALSE(original.owner_ship_);
}

// =============================================================================
// 测试 6: 边界条件和异常情况
// =============================================================================

TEST_F(TensorViewTest, ZeroSize)
{
  // 测试零大小构造：size=0 时应创建空的 TensorView
  const TensorView<float> tv(0);
  EXPECT_EQ(tv.size_, 0);
  EXPECT_FALSE(tv.owner_ship_); // size=0 时不拥有所有权
  EXPECT_EQ(tv.data_, nullptr); // 不分配内存
}

TEST_F(TensorViewTest, DefaultConstructorCreatesEmptyTensorView)
{
  // 测试默认构造函数创建空 TensorView
  TensorView<float> tv;

  EXPECT_EQ(tv.data(), nullptr);
  EXPECT_EQ(tv.size_, 0);
  EXPECT_FALSE(tv.owner_ship_);
}

TEST_F(TensorViewTest, CopyEmptyTensorView)
{
  // 测试拷贝空 TensorView
  const TensorView<float> empty; // 默认构造的空 TensorView
  TensorView copy(empty);

  EXPECT_EQ(copy.data(), nullptr);
  EXPECT_EQ(copy.size_, 0);
  EXPECT_FALSE(copy.owner_ship_);
}

TEST_F(TensorViewTest, MoveEmptyTensorView)
{
  // 测试移动空 TensorView
  TensorView<float> empty; // 默认构造的空 TensorView
  TensorView moved(std::move(empty));

  EXPECT_EQ(moved.data(), nullptr);
  EXPECT_EQ(moved.size_, 0);
  EXPECT_FALSE(moved.owner_ship_);

  // 原对象应保持空状态
  EXPECT_EQ(empty.data(), nullptr);
  EXPECT_EQ(empty.size_, 0);
  EXPECT_FALSE(empty.owner_ship_);
}

TEST_F(TensorViewTest, LargeSize)
{
  // 测试大尺寸分配
  constexpr std::size_t large_size = 1000000;
  const TensorView<float> tv(large_size);
  EXPECT_EQ(tv.size_, large_size);
  EXPECT_NE(tv.data_, nullptr);
}

TEST_F(TensorViewTest, NonEmptyDataPointer)
{
  // 测试非空数据指针
  float data[100] = {0};
  TensorView tv(data, 100);
  EXPECT_EQ(tv.data(), data);
  EXPECT_FALSE(tv.owner_ship_);
}

// =============================================================================
// 测试 7: 数据一致性
// =============================================================================

TEST_F(TensorViewTest, DataConsistencyAfterCopy)
{
  // 测试拷贝后的数据一致性
  constexpr std::size_t size = 100;
  std::vector<float> expected(size);
  for (std::size_t i = 0; i < size; ++i)
    expected[i] = static_cast<float>(i * 2);

  TensorView<float> original(size);
  std::ranges::copy(expected, original.data());

  TensorView<float> copy(original);

  for (std::size_t i = 0; i < size; ++i)
    EXPECT_FLOAT_EQ(copy.data()[i], expected[i]);
}

TEST_F(TensorViewTest, DataConsistencyAfterMove)
{
  // 测试移动后的数据一致性
  constexpr std::size_t size = 100;
  std::vector<float> expected(size);
  for (std::size_t i = 0; i < size; ++i)
    expected[i] = static_cast<float>(i * 3);

  TensorView<float> original(size);
  std::ranges::copy(expected, original.data());

  TensorView moved(std::move(original));

  for (std::size_t i = 0; i < size; ++i)
    EXPECT_FLOAT_EQ(moved.data()[i], expected[i]);
}

// =============================================================================
// 测试 8: 不同类型支持
// =============================================================================

TEST_F(TensorViewTest, FloatType)
{
  const TensorView<float> tv(10);
  EXPECT_EQ(tv.size_, 10);
}

TEST_F(TensorViewTest, DoubleType)
{
  const TensorView<double> tv(10);
  EXPECT_EQ(tv.size_, 10);
}

TEST_F(TensorViewTest, IntType)
{
  const TensorView<int> tv(10);
  EXPECT_EQ(tv.size_, 10);
}

TEST_F(TensorViewTest, UInt8Type)
{
  const TensorView<uint8_t> tv(10);
  EXPECT_EQ(tv.size_, 10);
}
