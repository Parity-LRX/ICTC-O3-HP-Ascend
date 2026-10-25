# Plain host LAMMPS with one persistent eager NPU model per MPI process.
# The interpreter loads torch_npu and Ascend C; no CUDA or Kokkos dependency.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
target_link_libraries(lammps PRIVATE Python3::Python)
target_compile_definitions(lammps PRIVATE MFF_ASCEND_EAGER=1)
target_compile_definitions(lammps PRIVATE MFF_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
