find_package(glfw3 3.3 REQUIRED)
find_package(glad REQUIRED)
find_package(fmt REQUIRED)
find_package(PNG REQUIRED)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/graphics
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/graphics/shaders
)

target_link_libraries(my_libraries PUBLIC glfw glad::glad PNG::PNG fmt::fmt TBB::tbb)

add_executable(freebirds main.cpp)
target_link_libraries(freebirds my_libraries fmt::fmt)