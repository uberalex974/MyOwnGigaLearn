cmake_minimum_required (VERSION 3.8)

project("GigaLearnBot")

# Create training executable (default)
file(GLOB_RECURSE TRAINING_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
list(REMOVE_ITEM TRAINING_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ExampleMainOptimized.cpp")
add_executable(GigaLearnBot ${TRAINING_SRC})

# Create separate RLBot deployment executable
file(GLOB_RECURSE DEPLOYMENT_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
list(REMOVE_ITEM DEPLOYMENT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ExampleMain.cpp")
add_executable(GigaLearnBot_Deploy ${DEPLOYMENT_SRC})

# Set C++ version to 20 for both executables
set_target_properties(GigaLearnBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD 20)
set_target_properties(GigaLearnBot_Deploy PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot_Deploy PROPERTIES CXX_STANDARD 20)

# Set RLBot deployment mode for deployment executable
target_compile_definitions(GigaLearnBot_Deploy PRIVATE RLBot_DEPLOYMENT=1)

# Make sure GigaLearnCPP is going to build in the same directory as us
# Otherwise, we won't be able to import it at runtime
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Include RLGymSim_PPO
add_subdirectory(GigaLearnCPP)
target_link_libraries(GigaLearnBot GigaLearnCPP)
target_link_libraries(GigaLearnBot_Deploy GigaLearnCPP)

# Include RLBot
add_subdirectory(RLBotCPP)
target_link_libraries(GigaLearnBot RLBotCPP)
target_link_libraries(GigaLearnBot_Deploy RLBotCPP)