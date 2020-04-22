//
// Created by human on 04.03.2020.
//

#ifndef OPENCL_CL_DEVICE_H
#define OPENCL_CL_DEVICE_H
#include <CL/cl.h>
typedef struct{
    cl_program* programs;
    cl_uint number_programs;
    cl_command_queue queue;
}cl_device;


typedef struct {
    cl_uint number_devices;
    cl_context context;
    cl_device* device_data;
    cl_device_id ids[0];
}cl_devices;

typedef struct {
    cl_uint number_platforms;
    cl_devices* devices;
    cl_platform_id ids[0];
}cl_platforms;

cl_platforms* cl_initDevices();
void cl_freeDevices(cl_platforms** platforms);
#endif //OPENCL_CL_DEVICE_H
