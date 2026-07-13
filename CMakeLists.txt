cmake_minimum_required(VERSION 3.16...3.25)

project(point_labeler VERSION 1.0 DESCRIPTION "point labeling of KITTI point clouds" LANGUAGES C CXX)

option(POINT_LABELER_BUILD_TESTS "Build point_labeler unit tests" ON)

set(OPENGL_VERSION "430" CACHE STRING "" FORCE)
set(ENABLE_NVIDIA_EXT On CACHE BOOL "" FORCE)

include(external/glow.cmake)

set(CMAKE_INCLUDE_CURRENT_DIR ON) # needs to be activated for qt generated files in build directory.
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
find_package(Qt5Xml REQUIRED)
find_package(Qt5Gui REQUIRED) 
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

# since now everything resides in "bin", we have to copy some stuff.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
foreach(OUTPUT_CONFIG Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${OUTPUT_CONFIG}" OUTPUT_CONFIG_UPPER)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUT_CONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
endforeach()

include_directories(src)

set(CMAKE_INCLUDE_CURRENT_DIR ON) # needs to be activated for qt generated files in build directory.
set(CMAKE_AUTOMOC ON)
add_compile_definitions($<$<CONFIG:Release>:NDEBUG>)
if(MSVC)
  add_compile_options(/W3 /permissive- /EHsc /bigobj)
else()
  add_compile_options(-Wall -UDEBUG_GL -UDEBUG)
endif()

# ensure that custom configuration file is not overwritten.
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/bin/settings.cfg")
  message("-- Copied default settings into bin. See 'settings.cfg' to change labeler settings.")
  configure_file(${CMAKE_SOURCE_DIR}/assets/settings.cfg ${CMAKE_SOURCE_DIR}/bin/settings.cfg COPYONLY)
endif()

# Cofiguration Files
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT WIN32)
  add_custom_target(link_labels ALL COMMAND ${CMAKE_COMMAND} -E create_symlink assets/labels.xml
                                            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/labels.xml COMMENT "Symlinking labels.xml to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
else()
  configure_file(assets/labels.xml ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/labels.xml COPYONLY)
  message(STATUS "Copied labels.xml to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()

add_subdirectory(src)

if(POINT_LABELER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(src/test)
endif()
