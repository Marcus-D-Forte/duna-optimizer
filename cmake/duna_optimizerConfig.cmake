include(CMakeFindDependencyMacro)
find_dependency(Eigen3)
find_dependency(OpenMP)
include(${CMAKE_CURRENT_LIST_DIR}/duna_optimizerTargets.cmake)