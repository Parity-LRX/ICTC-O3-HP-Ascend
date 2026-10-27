foreach(variable TORCH_INCLUDE_DIRS TORCH_LIBRARY_DIRS TORCH_NPU_PATH ICTC_CXX11_ABI)
    if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
        message(FATAL_ERROR "${variable} is required; configure with python -m ictc_o3.ascendc.build")
    endif()
endforeach()
if(NOT ICTC_CXX11_ABI MATCHES "^[01]$")
    message(FATAL_ERROR "ICTC_CXX11_ABI must match the PyTorch C++ ABI (0 or 1)")
endif()
if(NOT EXISTS "${TORCH_NPU_PATH}/include/torch_npu/csrc/core/npu/NPUStream.h")
    message(FATAL_ERROR "torch_npu development headers are missing from ${TORCH_NPU_PATH}")
endif()

add_library(ictc_torch_npu INTERFACE)
target_include_directories(ictc_torch_npu INTERFACE
    ${TORCH_INCLUDE_DIRS}
    "${TORCH_NPU_PATH}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform"
)
target_compile_definitions(ictc_torch_npu INTERFACE
    _GLIBCXX_USE_CXX11_ABI=${ICTC_CXX11_ABI}
)
target_link_directories(ictc_torch_npu INTERFACE
    ${TORCH_LIBRARY_DIRS} "${TORCH_NPU_PATH}/lib" "${ASCEND_CANN_PACKAGE_PATH}/lib64"
)
target_link_libraries(ictc_torch_npu INTERFACE
    torch torch_cpu c10 torch_npu ascendcl platform tiling_api register graph_base
)
