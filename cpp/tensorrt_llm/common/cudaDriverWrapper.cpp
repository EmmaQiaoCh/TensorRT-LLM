/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define CUDA_LIB_NAME "cuda"

#if defined(_WIN32)
#include <windows.h>
#define dllOpen(name) LoadLibrary("nv" name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) static_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name))
#else // For non-Windows platforms
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so.1", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/logger.h"
#include <cuda.h>

#include <cstdio>
#include <mutex>

namespace tensorrt_llm::common
{

std::shared_ptr<CUDADriverWrapper> CUDADriverWrapper::getInstance()
{
    static std::mutex mutex;
    static std::weak_ptr<CUDADriverWrapper> instance;
    std::shared_ptr<CUDADriverWrapper> result = instance.lock();
    if (result)
    {
        return result;
    }

    std::lock_guard<std::mutex> const lock(mutex);
    result = instance.lock();
    if (!result)
    {
        result = std::shared_ptr<CUDADriverWrapper>(new CUDADriverWrapper());
        instance = result;
    }
    return result;
}

CUDADriverWrapper::CUDADriverWrapper()
    : handle(dllOpen(CUDA_LIB_NAME))
{

    TLLM_CHECK_WITH_INFO(handle != nullptr, "CUDA driver library is not open correctly.");

    auto load_sym = [](void* handle, char const* name)
    {
        void* ret = dllGetSym(handle, name);
        return ret;
    };

    *reinterpret_cast<void**>(&_cuGetErrorName) = load_sym(handle, "cuGetErrorName");
    *reinterpret_cast<void**>(&_cuGetErrorString) = load_sym(handle, "cuGetErrorString");
    *reinterpret_cast<void**>(&_cuFuncSetAttribute) = load_sym(handle, "cuFuncSetAttribute");
    *reinterpret_cast<void**>(&_cuLinkComplete) = load_sym(handle, "cuLinkComplete");
    *reinterpret_cast<void**>(&_cuModuleUnload) = load_sym(handle, "cuModuleUnload");
    *reinterpret_cast<void**>(&_cuLinkDestroy) = load_sym(handle, "cuLinkDestroy");
    *reinterpret_cast<void**>(&_cuModuleLoadData) = load_sym(handle, "cuModuleLoadData");
    *reinterpret_cast<void**>(&_cuLinkCreate) = load_sym(handle, "cuLinkCreate_v2");
    *reinterpret_cast<void**>(&_cuModuleGetFunction) = load_sym(handle, "cuModuleGetFunction");
    *reinterpret_cast<void**>(&_cuModuleGetGlobal) = load_sym(handle, "cuModuleGetGlobal_v2");
    *reinterpret_cast<void**>(&_cuLibraryGetKernel) = load_sym(handle, "cuLibraryGetKernel");
    *reinterpret_cast<void**>(&_cuLibraryLoadData) = load_sym(handle, "cuLibraryLoadData");
    *reinterpret_cast<void**>(&_cuLibraryGetGlobal) = load_sym(handle, "cuLibraryGetGlobal");
    *reinterpret_cast<void**>(&_cuLibraryUnload) = load_sym(handle, "cuLibraryUnload");
    *reinterpret_cast<void**>(&_cuKernelSetAttribute) = load_sym(handle, "cuKernelSetAttribute");
    *reinterpret_cast<void**>(&_cuCtxGetDevice) = load_sym(handle, "cuCtxGetDevice");
    *reinterpret_cast<void**>(&_cuLinkAddFile) = load_sym(handle, "cuLinkAddFile_v2");
    *reinterpret_cast<void**>(&_cuLinkAddData) = load_sym(handle, "cuLinkAddData_v2");
    *reinterpret_cast<void**>(&_cuLaunchCooperativeKernel) = load_sym(handle, "cuLaunchCooperativeKernel");
    *reinterpret_cast<void**>(&_cuLaunchKernel) = load_sym(handle, "cuLaunchKernel");
    *reinterpret_cast<void**>(&_cuLaunchKernelEx) = load_sym(handle, "cuLaunchKernelEx");
    *reinterpret_cast<void**>(&_cuTensorMapEncodeTiled) = load_sym(handle, "cuTensorMapEncodeTiled");
    *reinterpret_cast<void**>(&_cuMemcpyDtoH) = load_sym(handle, "cuMemcpyDtoH_v2");
    *reinterpret_cast<void**>(&_cuDeviceGetAttribute) = load_sym(handle, "cuDeviceGetAttribute");
    *reinterpret_cast<void**>(&_cuOccupancyMaxActiveClusters) = load_sym(handle, "cuOccupancyMaxActiveClusters");
}

CUDADriverWrapper::~CUDADriverWrapper()
{
    dllClose(handle);
}

CUresult CUDADriverWrapper::cuGetErrorName(CUresult error, char const** pStr) const
{
    return (*_cuGetErrorName)(error, pStr);
}

CUresult CUDADriverWrapper::cuGetErrorString(CUresult error, char const** pStr) const
{
    return (*_cuGetErrorString)(error, pStr);
}

CUresult CUDADriverWrapper::cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) const
{
    return (*_cuFuncSetAttribute)(hfunc, attrib, value);
}

CUresult CUDADriverWrapper::cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) const
{
    return (*_cuLinkComplete)(state, cubinOut, sizeOut);
}

CUresult CUDADriverWrapper::cuModuleUnload(CUmodule hmod) const
{
    return (*_cuModuleUnload)(hmod);
}

CUresult CUDADriverWrapper::cuLinkDestroy(CUlinkState state) const
{
    return (*_cuLinkDestroy)(state);
}

CUresult CUDADriverWrapper::cuModuleLoadData(CUmodule* module, void const* image) const
{
    return (*_cuModuleLoadData)(module, image);
}

CUresult CUDADriverWrapper::cuLinkCreate(
    unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) const
{
    return (*_cuLinkCreate)(numOptions, options, optionValues, stateOut);
}

CUresult CUDADriverWrapper::cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, char const* name) const
{
    return (*_cuModuleGetFunction)(hfunc, hmod, name);
}

CUresult CUDADriverWrapper::cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, char const* name) const
{
    return (*_cuModuleGetGlobal)(dptr, bytes, hmod, name);
}

CUresult CUDADriverWrapper::cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, char const* name) const
{
    return (*_cuLibraryGetKernel)(pKernel, library, name);
}

CUresult CUDADriverWrapper::cuLibraryLoadData(CUlibrary* library, void const* code, CUjit_option* jitOptions,
    void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues,
    unsigned int numLibraryOptions) const
{
    return (*_cuLibraryLoadData)(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions,
        libraryOptionValues, numLibraryOptions);
}

CUresult CUDADriverWrapper::cuLibraryGetGlobal(
    CUdeviceptr* dptr, size_t* bytes, CUlibrary library, char const* name) const
{
    return (*_cuLibraryGetGlobal)(dptr, bytes, library, name);
}

CUresult CUDADriverWrapper::cuLibraryUnload(CUlibrary library) const
{
    return (*_cuLibraryUnload)(library);
}

CUresult CUDADriverWrapper::cuKernelSetAttribute(
    CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev) const
{
    return (*_cuKernelSetAttribute)(attrib, val, kernel, dev);
}

CUresult CUDADriverWrapper::cuCtxGetDevice(CUdevice* device) const
{
    return (*_cuCtxGetDevice)(device);
}

CUresult CUDADriverWrapper::cuLinkAddFile(CUlinkState state, CUjitInputType type, char const* path,
    unsigned int numOptions, CUjit_option* options, void** optionValues) const
{
    return (*_cuLinkAddFile)(state, type, path, numOptions, options, optionValues);
}

CUresult CUDADriverWrapper::cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size,
    char const* name, unsigned int numOptions, CUjit_option* options, void** optionValues) const
{
    return (*_cuLinkAddData)(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult CUDADriverWrapper::cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) const
{
    return (*_cuLaunchCooperativeKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult CUDADriverWrapper::cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) const
{
    return (*_cuLaunchKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

namespace
{
std::string stringify_launch_config(CUlaunchConfig const& config)
{
    std::stringstream ss;

    // Grid dimensions (Driver API uses separate fields)
    ss << "Grid Dimensions: (" << config.gridDimX << ", " << config.gridDimY << ", " << config.gridDimZ << ")\n";

    // Block dimensions
    ss << "Block Dimensions: (" << config.blockDimX << ", " << config.blockDimY << ", " << config.blockDimZ << ")\n";

    // Shared memory and stream (Driver API uses hStream)
    ss << "Shared Memory: " << config.sharedMemBytes << " bytes\n";
    ss << "Stream: " << (config.hStream ? "Custom" : "Default") << " (0x" << std::hex
       << reinterpret_cast<uintptr_t>(config.hStream) << ")\n";

    // Attributes (Driver API uses value instead of val)
    ss << "Attributes (" << config.numAttrs << "):\n";
    for (uint i = 0; i < config.numAttrs; ++i)
    {
        CUlaunchAttribute const& attr = config.attrs[i];
        ss << "  [" << i << "] ";

        switch (attr.id)
        {
        case CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION:
            ss << "Cluster Dimension: (" << attr.value.clusterDim.x << ", " << attr.value.clusterDim.y << ", "
               << attr.value.clusterDim.z << ")";
            break;

        case CU_LAUNCH_ATTRIBUTE_PRIORITY: ss << "Priority: " << attr.value.priority; break;

        // Handle other Driver API attributes here
        default: ss << "Unknown Attribute (ID=" << attr.id << ")"; break;
        }
        ss << "\n";
    }

    return ss.str();
}
} // namespace

CUresult CUDADriverWrapper::cuLaunchKernelEx(
    CUlaunchConfig const* config, CUfunction f, void** kernelParams, void** extra) const
{

    TLLM_LOG_DEBUG("Launch config: %s", stringify_launch_config(*config).c_str());
    TLLM_CHECK_DEBUG_WITH_INFO(
        (extra != nullptr) != (kernelParams != nullptr), "Exactly one of 'extra' and 'kernelParams' should be set.");
    return (*_cuLaunchKernelEx)(config, f, kernelParams, extra);
}

CUresult CUDADriverWrapper::cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void* globalAddress, cuuint64_t const* globalDim, cuuint64_t const* globalStrides,
    cuuint32_t const* boxDim, cuuint32_t const* elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) const
{
    return (*_cuTensorMapEncodeTiled)(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill);
}

CUresult CUDADriverWrapper::cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) const
{
    return (*_cuMemcpyDtoH)(dstHost, srcDevice, ByteCount);
}

CUresult CUDADriverWrapper::cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) const
{
    return (*_cuDeviceGetAttribute)(pi, attrib, dev);
}

CUresult CUDADriverWrapper::cuOccupancyMaxActiveClusters(
    int* maxActiveClusters, CUfunction f, CUlaunchConfig const* config) const
{
    return (*_cuOccupancyMaxActiveClusters)(maxActiveClusters, f, config);
}

} // namespace tensorrt_llm::common
