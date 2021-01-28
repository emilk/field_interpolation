cmake_minimum_required(VERSION 3.1)
project(field_interpolation)

set(EIGEN_INCLUDE_PATH "" CACHE STRING "Eigen include dir")
set(LOGURU_DIR "" CACHE STRING "loguru directory")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

include_directories(${LOGURU_DIR}/include ${EIGEN_INCLUDE_PATH})

add_library(field_interpolation STATIC
    field_interpolation/field_interpolation.cpp
    field_interpolation/sparse_linear.cpp)

target_compile_options(field_interpolation PRIVATE "-Wno-ignored-attributes -Wno-deprecated-declarations")

install(
    TARGETS field_interpolation
    ARCHIVE DESTINATION lib ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION lib ${CMAKE_INSTALL_LIBDIR}
    COMPONENT library)
install(
    DIRECTORY field_interpolation/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/field_interpolation
    FILES_MATCHING PATTERN "*.hpp")

