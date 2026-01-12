#include "device/Array.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/DistributionSamplers.hpp"

#include <array>

template<typename T>
DeviceArray<T>::DeviceArray(int N, const T &initValue)
    : m_size(N)
    , m_initialValue(initValue)
    , m_host(new T[std::max(N, 1)])
    , m_device(nullptr)
{
    
}

template<typename T>
DeviceArray<T>::DeviceArray(DeviceArray<T> &&arr)
{
    *this = std::move(arr);
}

template<typename T>
DeviceArray<T> &DeviceArray<T>::operator=(DeviceArray<T> &&arr)
{
    this->m_size = arr.m_size;
    this->m_initialValue = arr.m_initialValue;
    this->m_host = arr.m_host;
    this->m_device = arr.m_device;

    arr.m_size = 0;
    arr.m_host = nullptr;
    arr.m_device = nullptr;

    return *this;
}

template<typename T>
DeviceArray<T>::~DeviceArray()
{
    delete[] m_host;
    deleteDeviceMemory();
}

template<typename T>
void DeviceArray<T>::reset()
{
    std::fill_n(m_host, m_size, m_initialValue);

    if (m_device)
    {
        cudaMemcpy(m_device, m_host, sizeof(T)*m_size, cudaMemcpyHostToDevice);
    }
}

template<typename T>
void DeviceArray<T>::ensureDeviceAllocation()
{
    if (!m_device) {
        cudaMalloc(&m_device, sizeof(T)*m_size);
        reset();
    }
}

template<typename T>
void DeviceArray<T>::updateDeviceData()
{
    if (!m_device) return;

    cudaMemcpy(m_device, m_host, sizeof(T)*m_size, cudaMemcpyHostToDevice);
}

template<typename T>
void DeviceArray<T>::updateHostData()
{
    if (!m_device) return;

    cudaMemcpy(m_host, m_device, sizeof(T)*m_size, cudaMemcpyDeviceToHost);
}

template<typename T>
void DeviceArray<T>::deleteDeviceMemory()
{
    cudaFree(m_device);
    m_device = nullptr;
}

template struct DeviceArray<Vec4>;
template struct DeviceArray<float>;
template struct DeviceArray<uint32_t>;
template struct DeviceArray<AliasEntry>;
template struct DeviceArray<std::array<PathEntry, 10>>;
