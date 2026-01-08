add_definitions(-DRESOURCES_FILES_DIR="${PROJECT_SOURCE_DIR}/res")

add_executable(superpoint_example superpoint.cc)
target_link_libraries(superpoint_example point)

add_executable(superglue_example superglue.cc)
target_link_libraries(superglue_example glue npycpp)

add_executable(pointglue_example pointglue.cc)
target_link_libraries(pointglue_example point glue)