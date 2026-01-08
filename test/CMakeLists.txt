find_package(GTest REQUIRED)


include_directories(${CMAKE_SOURCE_DIR}/include)

# 设置资源文件路径（相对于源码目录）
set(RESOURCES_DIR "${CMAKE_SOURCE_DIR}/res")

# 测试可执行文件列表
set(TEST_SOURCES
    test_tensorview.cpp
    test_superpoint.cpp
    test_superglue.cpp
)

# 为每个测试创建单独的可执行文件
foreach(test_source ${TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME_WE)
  add_executable(${test_name} ${test_source})
  target_link_libraries(${test_name} PRIVATE
      GTest::GTest
      GTest::Main
      point
      glue
      utils
      npycpp)

  target_compile_definitions(${test_name} PRIVATE $<$<CONFIG:Debug>:DEBUG>)
  target_compile_definitions(${test_name} PRIVATE RESOURCES_FILES_DIR="${RESOURCES_DIR}")

  add_test(NAME ${test_name} COMMAND ${test_name})
  set_tests_properties(${test_name} PROPERTIES LABELS "unit")
endforeach()

# 自定义测试目标
add_custom_target(test_units
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_SOURCES}
    COMMENT "Running unit tests..."
    USES_TERMINAL)

# 内存泄漏检测目标（使用 Valgrind）
add_custom_target(test_memory
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    -T MemCheck
    DEPENDS ${TEST_SOURCES}
    COMMENT "Running memory leak tests..."
    USES_TERMINAL)

# 启用内存检查
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
  set(CTEST_MEMORYCHECK_COMMAND ${VALGRIND_EXECUTABLE})
  set(CTEST_MEMORYCHECK_COMMAND_OPTIONS
      "--leak-check=full --show-leak-kinds=all --track-origins=yes")
endif()