# Benchmark CMakeLists.txt

# 设置资源文件路径
set(RESOURCES_DIR "${CMAKE_SOURCE_DIR}/res")

# 性能测试源文件列表
set(BENCHMARK_SOURCES
    benchmark_superpoint.cc
    benchmark_pointglue.cc
)

# 为每个性能测试创建可执行文件
foreach(benchmark_source ${BENCHMARK_SOURCES})
  get_filename_component(benchmark_name ${benchmark_source} NAME_WE)
  add_executable(${benchmark_name} ${benchmark_source})

  target_link_libraries(${benchmark_name} PRIVATE
      benchmark::benchmark
      benchmark::benchmark_main
      point
      glue
      utils
      npycpp)

  target_compile_definitions(${benchmark_name} PRIVATE $<$<CONFIG:Debug>:DEBUG>)
  target_compile_definitions(${benchmark_name} PRIVATE RESOURCES_FILES_DIR="${RESOURCES_DIR}")

  target_include_directories(${benchmark_name} PRIVATE
      ${CMAKE_SOURCE_DIR}/include)
endforeach()

# 自定义性能测试目标
add_custom_target(run_benchmarks
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${BENCHMARK_SOURCES}
    COMMENT "Running benchmarks..."
    USES_TERMINAL)