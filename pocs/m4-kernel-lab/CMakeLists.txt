cmake_minimum_required(VERSION 3.16)
project(m4_kernel_lab C CXX)

# FASE 4B microbenchmark: independent build, no llama.cpp / ggml linkage yet.

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# On Apple Silicon enable NEON-friendly tuning. ARM macOS already has NEON.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    add_compile_options(-mcpu=native)
endif()

add_executable(m4_kernel_lab
    main.cpp
    kernels/kernel_scalar.c
    kernels/kernel_scalar_opt.c
    kernels/kernel_ggml.c
)

target_include_directories(m4_kernel_lab PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link the ggml-cpu dylib from the main llama.cpp build so we can compare
# against its production Q4_K x Q8_K kernel.
# Override LLAMA_BUILD_DIR if your build lives somewhere other than ../../build.
set(LLAMA_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build" CACHE PATH "llama.cpp build dir")

find_library(GGML_CPU_LIB
    NAMES ggml-cpu
    HINTS "${LLAMA_BUILD_DIR}/bin"
    NO_DEFAULT_PATH)
find_library(GGML_BASE_LIB
    NAMES ggml-base
    HINTS "${LLAMA_BUILD_DIR}/bin"
    NO_DEFAULT_PATH)

if(GGML_CPU_LIB AND GGML_BASE_LIB)
    message(STATUS "ggml-cpu : ${GGML_CPU_LIB}")
    message(STATUS "ggml-base: ${GGML_BASE_LIB}")
    target_link_libraries(m4_kernel_lab PRIVATE ${GGML_CPU_LIB} ${GGML_BASE_LIB})
    target_compile_definitions(m4_kernel_lab PRIVATE M4_KERNEL_LAB_HAVE_GGML=1)

    # macOS dylibs use install_name relative to the build dir; tell the loader
    # where to find them at runtime so the lab works without `make install`.
    if(APPLE)
        set_target_properties(m4_kernel_lab PROPERTIES
            BUILD_RPATH   "${LLAMA_BUILD_DIR}/bin"
            INSTALL_RPATH "${LLAMA_BUILD_DIR}/bin")
    endif()
else()
    message(WARNING "ggml-cpu/ggml-base not found in ${LLAMA_BUILD_DIR}/bin -- "
                    "build the main llama.cpp first or set -DLLAMA_BUILD_DIR=...")
endif()
