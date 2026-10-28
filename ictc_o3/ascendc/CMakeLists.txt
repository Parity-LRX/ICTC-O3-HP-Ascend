cmake_minimum_required(VERSION 3.20)

# Discover the installed CANN ASC language before project() enables it.
include(cmake/AscendC.cmake)
project(ictc_ascendc VERSION 0.1.0 LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(GNUInstallDirs)
include(cmake/TorchNPU.cmake)

ascendc_library(ictc_ascendc_kernel STATIC
    op_kernel/electrostatics.cpp
    op_kernel/vector.cpp
    op_kernel/weighted_map.cpp
    op_kernel/epilogue.cpp
    op_kernel/epilogue_adjoint.cpp
    op_kernel/batched_vector.cpp
    op_kernel/matmul.cpp
)
target_include_directories(ictc_ascendc_kernel PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" ${ICTC_HOST_CXX_INCLUDES}
)
target_compile_options(ictc_ascendc_kernel PRIVATE
    --npu-soc=${SOC_VERSION} -std=c++17
    $<$<CONFIG:Release>:-O2>
    "$<$<CONFIG:RelWithDebInfo>:-O2;-g>"
)
target_compile_definitions(ictc_ascendc_kernel PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${ICTC_CXX11_ABI}
)

add_library(ictc_ascendc SHARED
    op_host/electrostatics_bindings.cpp
    op_host/bindings.cpp
    op_host/epilogue_bindings.cpp
    op_host/epilogue_adjoint_bindings.cpp
    op_host/batched_vector_bindings.cpp
    op_host/matmul_bindings.cpp
)
target_include_directories(ictc_ascendc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(ictc_ascendc PRIVATE ictc_ascendc_kernel ictc_torch_npu)
set_target_properties(ictc_ascendc PROPERTIES
    BUILD_RPATH "${TORCH_LIBRARY_DIRS};${TORCH_NPU_PATH}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64"
    INSTALL_RPATH "${TORCH_LIBRARY_DIRS};${TORCH_NPU_PATH}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64"
)
install(TARGETS ictc_ascendc
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT ictc_runtime
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ictc_ascendc_build.json"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT ictc_runtime OPTIONAL
)
