# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ljl/ljl_test/llama/ggml/src/ggml-vulkan/vulkan-shaders"
  "/home/ljl/ljl_test/llama/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build"
  "/home/ljl/ljl_test/llama/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix"
  "/home/ljl/ljl_test/llama/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/tmp"
  "/home/ljl/ljl_test/llama/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp"
  "/home/ljl/ljl_test/llama/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src"
  "/home/ljl/ljl_test/llama/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ljl/ljl_test/llama/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ljl/ljl_test/llama/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp${cfgdir}") # cfgdir has leading slash
endif()
