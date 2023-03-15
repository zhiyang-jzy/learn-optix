cmake_minimum_required(VERSION 3.24)
project(learnoptix CXX  CUDA C)

find_package(CUDAToolkit)
find_package(CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
find_package(OptiX REQUIRED)

include_directories(${OptiX_INCLUDE})

set(CMAKE_CUDA_STANDARD 17)




macro(cuda_compile_and_embed output_var cuda_file)
    set(c_var_name ${output_var})
    cuda_compile_ptx(ptx_files ${cuda_file} OPTIONS --generate-line-info -use_fast_math --keep)
    list(GET ptx_files 0 ptx_file)
    set(embedded_file ${ptx_file}_embedded.c)
    message("adding rule to compile and embed ${cuda_file} to \"const char ${var_name}[];\"")
    message(${embedded_file})
    add_custom_command(
            OUTPUT ${embedded_file}
            COMMAND bin2c -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
            DEPENDS ${ptx_file}
            COMMENT "compiling (and embedding ptx from) ${cuda_file}"
    )
    set(${output_var} ${embedded_file})
endmacro()

cuda_compile_and_embed(embedded_ptx_code device_program.cu)

message(${embedded_ptx_code})
add_executable(learnoptix ${embedded_ptx_code}  LauchParams.h main.cpp cuda_buffer.cpp cuda_buffer.h common.h common.h stb_image_write.h)

target_link_libraries(learnoptix PRIVATE CUDA::cudart ${optix_LIBRARY} CUDA::cuda_driver)

#set_target_properties(learnoptix PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

