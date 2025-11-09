#include "device/Array.hpp"
#include "tracing/Intersection.hpp"

#include <array>

template<typename T>
DeviceArray<T>::DeviceArray(int N, const T &initValue)
    : m_size(N)
    , m_initialValue(initValue)
    , m_host(new T[N])
    , m_device(nullptr)
{
    
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
template struct DeviceArray<uint32_t>;
template struct DeviceArray<std::array<PathEntry, 10>>;
