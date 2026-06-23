#pragma once

#include "Utils.hpp"
#include <span>

template<typename T>
struct DeviceArray
{
    DeviceArray() : DeviceArray(0, T{}) {}
    DeviceArray(int N, const T &initValue);
    DeviceArray(DeviceArray &&arr);
    DeviceArray &operator=(DeviceArray &&arr);

    DeviceArray(const DeviceArray &arr) = delete;
    DeviceArray &operator=(const DeviceArray &arr) = delete;

    ~DeviceArray();

    void reset();
    void deleteDeviceMemory();
    void ensureDeviceAllocation();
    void updateDeviceData();
    void updateHostData();

    T *devicePtr() { return m_device; }
    T *hostPtr() { return m_host; }
    std::span<T> hostSpan() { return std::span{m_host, size()}; }
    const T *hostPtr() const { return m_host; }
    size_t size() const { return static_cast<size_t>(m_size); }

    int m_size;
    T m_initialValue;
    T *m_host;
    T *m_device;
};



namespace device_array
{
    void *deviceAlloc(int byteCount);
    void deviceFree(void *ptr);

    void copyToDevice(void *device_target, const void *host_source, int byteCount);
    void copyToHost(void *host_target, const void *device_source, int byteCount);
}

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

    updateDeviceData();
}

template<typename T>
void DeviceArray<T>::ensureDeviceAllocation()
{
    if (!m_device) {
        m_device = static_cast<T*>(device_array::deviceAlloc(sizeof(T)*m_size));
        updateDeviceData();
    }
}

template<typename T>
void DeviceArray<T>::updateDeviceData()
{
    if (!m_device) return;

    device_array::copyToDevice(m_device, m_host, sizeof(T)*m_size);
}

template<typename T>
void DeviceArray<T>::updateHostData()
{
    if (!m_device) return;

    device_array::copyToHost(m_host, m_device, sizeof(T)*m_size);
}

template<typename T>
void DeviceArray<T>::deleteDeviceMemory()
{
    if (m_device) {
        device_array::deviceFree(m_device);
        m_device = nullptr;
    }
}