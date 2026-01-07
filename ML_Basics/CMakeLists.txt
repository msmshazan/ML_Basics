# CMakeList.txt : CMake project for ML_Basics, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (ML_Basics "main.c" "main.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ML_Basics PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
