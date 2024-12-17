find_package(TBB REQUIRED)

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

target_link_libraries(my_libraries PUBLIC TBB::tbb)

add_executable(freebirds main.cpp)
target_link_libraries(freebirds my_libraries)