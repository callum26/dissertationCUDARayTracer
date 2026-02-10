# minimum cmake versoin
cmake_minimum_required(VERSION 3.10)

# project name
project(rayTracer)

# setting the C++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# libraries --

# glad
set(GLAD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/import/glad/src/glad.c")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/import/glad/include")

# glfw
add_subdirectory(import/glfw-3.4)

# executable --
add_executable(rayTracer main.cpp ${GLAD_SRC})

# linking libraries
target_link_libraries(rayTracer PRIVATE glfw ${CMAKE_THREAD_LIBS_INIT})