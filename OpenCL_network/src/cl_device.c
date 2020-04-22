//
// Created by human on 04.03.2020.
//
#include "cl_device.h"
#include <stdio.h>
#include <string.h>
#define ARRAY_SIZE(array) (sizeof(array) / sizeof(*array))

struct cl_devices* cl_allocateDevices(cl_uint number_devices){
    cl_devices* devices = (cl_devices*)calloc(number_devices * sizeof(cl_device_id) + sizeof(cl_devices), 1);
    devices->number_devices = number_devices;
    devices->device_data = (cl_device*)calloc(number_devices, sizeof(cl_device));
    return devices;
}
cl_platforms* cl_allocatePlatforms(cl_uint number_platforms){
    cl_platforms* platforms = (struct cl_devices*)malloc(number_platforms * sizeof(cl_platform_id) + sizeof(cl_platforms));
    platforms->number_platforms = number_platforms;
    platforms->devices = (cl_devices*)calloc(number_platforms, sizeof(cl_devices));
    return platforms;
}

void printPlatformInfo(cl_platform_id* platform, cl_uint number_platforms){
    char* info = NULL;
    size_t size_info = 0;
    cl_platform_info info_type[] = { CL_PLATFORM_PROFILE, CL_PLATFORM_VERSION, CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_EXTENSIONS };
    for (cl_uint i = 0; i < number_platforms; i++){
        for (cl_uint j = 0; j < ARRAY_SIZE(info_type); j++){
            clGetPlatformInfo(platform[i], info_type[j], 0, NULL, &size_info);
            info = realloc(info, size_info);
            clGetPlatformInfo(platform[i], info_type[j], size_info, info, NULL);
            printf("CL_PLATFORM : %s\n", info);
        }
    }
    if (info) free(info);
}
void printDeviceInfo(cl_device_id* device, cl_uint number_devices){
    cl_uint type;

    for (cl_uint i = 0; i < number_devices; i++){
        clGetDeviceInfo(device[i], CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &type, 0);
        printf("CL_DEVICE_VENDOR_ID : %u\n", type);
        clGetDeviceInfo(device[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &type, 0);
        printf("CL_DEVICE_MAX_COMPUTE_UNITS : %u\n", type);
        clGetDeviceInfo(device[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &type, 0);
        printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : %u\n", type);
    }
}
void printContextInfo(cl_context* context){
    cl_uint type;
    clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &type, 0);
    printf("CL_CONTEXT_REFERENCE_COUNT : %u\n", type);

}

cl_platforms* cl_initDevices(){
    cl_platforms* platforms = NULL;
    cl_int error_code;
    cl_uint m_number_devices = 0;
    cl_uint m_number_platforms = 0;

    clGetPlatformIDs(0, NULL, &m_number_platforms);
    platforms = cl_allocatePlatforms(m_number_platforms);
    clGetPlatformIDs(platforms->number_platforms, platforms->ids, NULL);
    printPlatformInfo(platforms->ids, platforms->number_platforms);

    cl_device_info device_info;
    cl_command_queue_properties queue_properties = CL_QUEUE_PROFILING_ENABLE;
    for (cl_uint m_i = 0; m_i < platforms->number_platforms; m_i++){

        clGetDeviceIDs(platforms->ids[m_i], CL_DEVICE_TYPE_ALL, 0, NULL, (cl_uint*)&m_number_devices);
        platforms[m_i].devices = cl_allocateDevices(m_number_devices);
        clGetDeviceIDs(platforms->ids[m_i], CL_DEVICE_TYPE_ALL, platforms[m_i].devices->number_devices, platforms[m_i].devices->ids, NULL);
        printDeviceInfo(platforms[m_i].devices->ids, platforms[m_i].devices->number_devices);

        platforms[m_i].devices->context = clCreateContext(NULL, platforms[m_i].devices->number_devices, platforms[m_i].devices->ids, NULL, NULL, &error_code);
        if (error_code != 0){
            printf("Error clCreateContextFromType!\n");
            exit(1);
        }

        for (cl_uint i = 0; i < platforms[m_i].devices->number_devices; i++){
            platforms[m_i].devices->device_data[i].queue = clCreateCommandQueue(platforms[m_i].devices->context, platforms[m_i].devices->ids[i], queue_properties, &error_code);
            if (error_code != 0){
                printf("Error clCreateCommandQueue!\n");
                exit(1);
            }
        }
    }
    return platforms;
}
void cl_freeDevices(cl_platforms** platforms){
    if (platforms[0])
    for (cl_uint m_i = 0; m_i < platforms[0]->number_platforms; m_i++) {
        if (platforms[0][m_i].devices){
            for (cl_uint i = 0; i < platforms[0][m_i].devices->number_devices; i++) {
                for (cl_uint j = 0; j < platforms[0][m_i].devices->device_data[i].number_programs; j++) {
                    clReleaseProgram(platforms[0][m_i].devices->device_data[i].programs[j]);
                }
                if (platforms[0][m_i].devices->device_data[i].programs) free(platforms[0][m_i].devices->device_data[i].programs);
                clReleaseCommandQueue(platforms[0][m_i].devices->device_data[i].queue);
                clReleaseDevice(platforms[0][m_i].devices->ids[i]);
                clReleaseContext(platforms[0][m_i].devices[i].context);
            }
            if (platforms[0][m_i].devices->device_data) free(platforms[0][m_i].devices->device_data);
            if (platforms[0][m_i].devices) free(platforms[0][m_i].devices);
        }
    }
    if (platforms[0]) free(platforms[0]);
    platforms[0] = NULL;
}