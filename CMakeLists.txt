# 9/9/2015 v0.1

cmake_minimum_required (VERSION 2.6)
project (Maze_Experiment)

set(BOOST_NUMPY_LIBRARY ~/Desktop/Boost.NumPy/build/lib/libboost_numpy.so)
set(BOOST_NUMPY_INCLUDE_DIR ~/Desktop/Boost.NumPy/boost)

find_package(Boost COMPONENTS python REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
 
include_directories(.)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${BOOST_PYTHON_INCLUDE_DIR})
include_directories(${BOOST_NUMPY_INCLUDE_DIR})

# set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -g")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")

add_library(maze_lib SHARED experiment_main.cpp maze.cpp)
set_target_properties(maze_lib PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(maze_lib ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${BOOST_NUMPY_LIBRARY} -lrt)

add_executable(maze_exec experiment_main.cpp maze.cpp)
target_link_libraries(maze_exec ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${BOOST_NUMPY_LIBRARY} -lrt)

set(CMAKE_INSTALL_PREFIX ".")
install(TARGETS maze_lib DESTINATION "..")
