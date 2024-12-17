# Check if TBB_ROOT is set
if(NOT DEFINED ENV{TBB_ROOT})
    message(FATAL_ERROR "Environment variable TBB_ROOT is not set. Please set it before running CMake.")
endif()

# Define TBB paths
set(TBB_INCLUDE_PATH "$ENV{TBB_ROOT}/include" CACHE PATH "Path to TBB include directory" FORCE)
set(TBB_LIBRARY_PATH "$ENV{TBB_ROOT}/lib" CACHE PATH "Path to TBB library directory" FORCE)

# Include TBB
find_package(TBB REQUIRED)

# Find all .cpp files in the source directory
file(GLOB_RECURSE ALL_CPP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Exclude `main.cpp` from the list of source files
list(REMOVE_ITEM ALL_CPP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

# Create a library from the remaining .cpp files
add_library(my_libraries STATIC ${ALL_CPP_FILES})

# Include directories for the library
target_include_directories(my_libraries PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/laws
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/agents
    ${TBB_INCLUDE_PATH} # Add TBB include path
)

# Link TBB to the library
target_link_libraries(my_libraries PUBLIC TBB::tbb)

# Define the executable
add_executable(freebirds main.cpp)

# Link the library to the executable
target_link_libraries(freebirds PUBLIC my_libraries)
