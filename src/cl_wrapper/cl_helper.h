#pragma once
#include <vector>
#include "cl/cl.h"
#include <memory>
#include <string>

namespace cl_wrapper {

    class DeviceDetail {

    public:
        const cl_device_id device_id;
        const cl_platform_id platform_id;
        const std::string device_name;
        const size_t max_local_group_size;

        DeviceDetail(cl_device_id device_id, cl_platform_id platform_id, size_t max_local_group_size, std::string&& name) : device_id(device_id), platform_id(platform_id), device_name(std::move(name)), max_local_group_size(max_local_group_size) {

        }

    };

    class PlatformDetail {

    public:
        const cl_platform_id platform_id;
        const std::string platform_name;
        const std::string opencl_version;
        const std::vector<DeviceDetail> devices;

        PlatformDetail(cl_platform_id id, std::string&& name, std::string&& version, std::vector<DeviceDetail>&& devices) : platform_id(id), platform_name(std::move(name)), opencl_version(version), devices(std::move(devices)) {

        }

        std::string to_string() {
            std::string res = "Platform: " + platform_name + "\n";
            for (int i = 0; i < devices.size(); i++) {
                res += "Device " + std::to_string(i) + ": " + devices[i].device_name + "\n";
            }
            res += "-------------------------";

            return res;
        }
    };

    class Context {
    public:
        std::unique_ptr<cl_context> context;

        Context(const Context& ctx) = delete;
        Context() = delete;

        Context(Context&& ctx) {
            context = std::move(ctx.context);
        }

        Context(std::unique_ptr<cl_context>&& ctx) : context(std::move(ctx)) {

        }
    };

    std::vector<PlatformDetail> get_platforms();
    std::vector<DeviceDetail> get_devices(cl_platform_id id);

    PlatformDetail get_platform_details(cl_platform_id id);
    std::vector<cl_platform_id> get_platforms_ids();

    std::vector<cl_device_id> get_device_ids(cl_platform_id id);
    DeviceDetail get_device_details(cl_platform_id platform_id, cl_device_id id);

    Context create_context(const DeviceDetail& device);

    std::vector<cl_platform_id> cl_wrapper::get_platforms_ids() {
        const cl_uint MAX_PLATFORMS = 10;

        cl_uint actual_platform_count = 0;
        cl_platform_id id_array[10];

        clGetPlatformIDs(MAX_PLATFORMS, id_array, &actual_platform_count);

        std::vector<cl_platform_id> res;

        for (int i = 0; i < actual_platform_count; i++) {
            res.push_back(id_array[i]);
        }

        return res;
    }


    cl_wrapper::PlatformDetail cl_wrapper::get_platform_details(cl_platform_id id) {
        const size_t MAX_BUFFER_SIZE = 100;

        char name[MAX_BUFFER_SIZE];
        char vendor[MAX_BUFFER_SIZE];
        char version[MAX_BUFFER_SIZE];

        size_t name_size;
        size_t vendor_size;
        size_t version_size;

        clGetPlatformInfo(id, CL_PLATFORM_NAME, MAX_BUFFER_SIZE, name, &name_size);
        clGetPlatformInfo(id, CL_PLATFORM_VERSION, MAX_BUFFER_SIZE, version, &version_size);
        clGetPlatformInfo(id, CL_PLATFORM_VENDOR, MAX_BUFFER_SIZE, vendor, &vendor_size);

        return PlatformDetail(id, std::string(name, name_size), std::string(version, version_size), get_devices(id));
    }

    std::vector<cl_wrapper::PlatformDetail> cl_wrapper::get_platforms() {
        std::vector<PlatformDetail> details;

        auto ids = get_platforms_ids();

        for (int i = 0; i < ids.size(); i++) {
            details.push_back(get_platform_details(ids[i]));
        }

        return details;
    }


    std::vector<cl_wrapper::DeviceDetail> cl_wrapper::get_devices(cl_platform_id id) {
        auto devices = get_device_ids(id);
        std::vector<DeviceDetail> details;

        for (int i = 0; i < devices.size(); i++) {
            details.push_back(get_device_details(id, devices[i]));
        }

        return details;
    }

    cl_wrapper::Context cl_wrapper::create_context(const cl_wrapper::DeviceDetail& device) {
        return cl_wrapper::Context(std::make_unique<cl_context>(clCreateContext(nullptr, 1, &device.device_id, nullptr, nullptr, nullptr)));
    }

    std::vector<cl_device_id> cl_wrapper::get_device_ids(cl_platform_id id) {
        const size_t MAX_DEVICES = 100;
        cl_device_id ids[MAX_DEVICES];
        cl_uint count;

        clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, MAX_DEVICES, ids, &count);

        std::vector<cl_device_id> devices;

        for (int i = 0; i < count; i++) {
            devices.push_back(ids[i]);
        }

        return devices;
    }

    cl_wrapper::DeviceDetail cl_wrapper::get_device_details(cl_platform_id platform_id, cl_device_id id) {
        const size_t MAX_BUFFER_SIZE = 100;

        char name[MAX_BUFFER_SIZE];
        size_t name_size;

        cl_uint max_local_group_size;

        void* group_size_buffer[sizeof(size_t)];

        clGetDeviceInfo(id, CL_DEVICE_NAME, MAX_BUFFER_SIZE, name, &name_size);
        clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), group_size_buffer, nullptr);

        return DeviceDetail(id, platform_id, *((size_t*)group_size_buffer), std::move(std::string(name, name_size)));
    }
}