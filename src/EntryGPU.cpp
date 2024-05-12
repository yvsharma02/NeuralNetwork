#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <iostream>
#include <vector>

#include "cl_wrapper/cl_helper.h"

//std::string kernel_code =
//    "   void kernel simple_add(global const int* A, global const int* B, global int* C){C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];}";
std::string kernel_code =
"   void kernel simple_add(global const int* A, global const int* B, global int* C){C[get_global_id(0)]=get_global_id(0);}";

int main() {
    auto platforms = cl_wrapper::get_platforms();
    auto context = cl_wrapper::create_context(platforms[0].devices[0]);
    //    std::cout << "SIZE: ";
    //    std::cout << platforms[0].devices[0].max_local_group_size << std::endl;
    const size_t ITEM_COUNT = 1024 * 1024;
    const size_t BUFFER_SIZE = ITEM_COUNT * sizeof(int);
    const size_t X = ITEM_COUNT;

    int* A_h = new int[ITEM_COUNT];
    int* B_h = new int[ITEM_COUNT];
    int* C_h = new int[ITEM_COUNT];

    for (int i = 0; i < ITEM_COUNT; i++) {
        A_h[i] = i;
        B_h[i] = X - i;
        C_h[i] = -2;
    }

    //    int A_h[] = { 2, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    //    int B_h[] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    //    int C_h[] = { 99, 99, 99, 99, 99, 99, 99, 99, 99, 99 };

    cl_int err;
    auto A_d = clCreateBuffer(*context.context, CL_MEM_READ_WRITE, BUFFER_SIZE, nullptr, &err);
    std::cout << err << std::endl;
    auto B_d = clCreateBuffer(*context.context, CL_MEM_READ_WRITE, BUFFER_SIZE, nullptr, &err);
    std::cout << err << std::endl;
    auto C_d = clCreateBuffer(*context.context, CL_MEM_READ_WRITE, BUFFER_SIZE, nullptr, &err);
    std::cout << err << std::endl;

    auto commandQueue = clCreateCommandQueue(*context.context, platforms[0].devices[0].device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    std::cout << err << std::endl;

    clEnqueueWriteBuffer(commandQueue, A_d, CL_TRUE, 0, BUFFER_SIZE, A_h, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(commandQueue, B_d, CL_TRUE, 0, BUFFER_SIZE, B_h, 0, nullptr, nullptr);
    //    clEnqueueWriteBuffer(commandQueue, C_d, CL_TRUE, 0, BUFFER_SIZE, C_h, 0, nullptr, nullptr);    

    //    auto buffer = clCreateBuffer(*context.context, CL_MEM_READ_WRITE, sizeof(int) * 10);

    const char* strings_arr[] = { kernel_code.c_str() };
    size_t lens_arr[] = { kernel_code.size() };

    auto program = clCreateProgramWithSource(*context.context, 1, strings_arr, lens_arr, nullptr);

    cl_int res = clBuildProgram(program, 1, &platforms[0].devices[0].device_id, nullptr, /*[](cl_program p, void* dat) -> void {
        std::cout << "Yay!" << std::endl;
    }*/ nullptr, nullptr);
    if (res == CL_SUCCESS) {
        std::cout << " A " << std::endl;
        auto kernal = clCreateKernel(program, "simple_add", nullptr);
        std::cout << " B " << std::endl;
        //        clEnqueueNDRangeKernel(commandQueue, kernal, 2, nullptr, );
        cl_event event;

        //        err = clEnqueueNDRangeKernel (commandQueue, kernal, 1, nullptr, x, y, 0, nullptr, &event);

        clSetKernelArg(kernal, 0, sizeof(void*), &A_d);
        clSetKernelArg(kernal, 1, sizeof(void*), &B_d);
        clSetKernelArg(kernal, 2, sizeof(void*), &C_d);

        const size_t global_groups_size = ITEM_COUNT;
        const size_t local_groups_size = ITEM_COUNT / 1024;
        //        err = clEnqueueTask (commandQueue, kernal, 0, nullptr, &event);
        err = clEnqueueNDRangeKernel(commandQueue, kernal, 1, nullptr, &global_groups_size, &local_groups_size, 0, nullptr, &event);
        std::cout << " C " << err << std::endl;
        clWaitForEvents(1, &event);

        std::cout << " D " << std::endl;
        int resss = clEnqueueReadBuffer(commandQueue, C_d, CL_TRUE, 0, BUFFER_SIZE, C_h, 0, nullptr, nullptr);

        std::cout << resss << std::endl;

        std::cout << " E " << std::endl;
        for (int i = 0; i < ITEM_COUNT; i++) {
            std::cout << C_h[i] << " ";
        }
        std::cout << std::endl;

    }
    else {
        std::cout << "Something has gone totally wrong :(" << std::endl;
    }

    delete[] A_h;
    delete[] B_h;
    delete[] C_h;

    return 0;
}