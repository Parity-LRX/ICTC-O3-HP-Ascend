cmake_minimum_required(VERSION 3.16)
project(ictc_ascendc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RUN_MODE "npu" CACHE STRING "Ascend C execution mode")
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

ascendc_library(ictc_ascendc_kernel STATIC csrc/vector.cpp csrc/weighted_map.cpp csrc/epilogue.cpp csrc/matmul.cpp)
ascendc_include_directories(ictc_ascendc_kernel PRIVATE ${ICTC_HOST_CXX_INCLUDES})
add_library(ictc_ascendc SHARED csrc/bindings.cpp csrc/epilogue_bindings.cpp csrc/matmul_bindings.cpp)
target_include_directories(ictc_ascendc PRIVATE ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform)
target_compile_definitions(ictc_ascendc PRIVATE _GLIBCXX_USE_CXX11_ABI=${ICTC_CXX11_ABI})
target_link_directories(ictc_ascendc PRIVATE ${TORCH_LIBRARY_DIRS} ${TORCH_NPU_PATH}/lib
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(ictc_ascendc PRIVATE ictc_ascendc_kernel torch torch_cpu c10
    torch_npu ascendcl platform tiling_api register)
set_target_properties(ictc_ascendc PROPERTIES BUILD_RPATH
    "${TORCH_LIBRARY_DIRS};${TORCH_NPU_PATH}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64")
