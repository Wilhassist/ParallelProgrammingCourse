# Check if TBB_ROOT is set
if(DEFINED ENV{TBB_ROOT})
    set(TBB_ROOT $ENV{TBB_ROOT})
    message(STATUS "Using TBB from: ${TBB_ROOT}")
else()
    message(FATAL_ERROR "TBB_ROOT environment variable not set!")
endif()

# Find TBB package
find_package(TBB REQUIRED PATHS ${TBB_ROOT}/cmake)

# You can now use TBB in your project
include_directories(${TBB_ROOT}/include)
link_directories(${TBB_ROOT}/lib)

file(GLOB_RECURSE ALL_CPP_FILES "*.cpp")

# List of files to exclude
list(REMOVE_ITEM ALL_CPP_FILES "main.cpp")

# Create a library from the remaining .cpp files
add_library(my_libraries ${ALL_CPP_FILES})


target_include_directories(my_libraries PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/laws
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/agents
)

target_link_libraries(my_libraries PUBLIC TBB::TBB)

add_executable(freebirds main.cpp)
target_link_libraries(freebirds my_libraries)