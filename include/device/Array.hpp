#pragma once

#include "Utils.hpp"

template<typename T>
struct DeviceArray
{
    DeviceArray() : DeviceArray(0, T{}) {}
    DeviceArray(int N, const T &initValue);
    DeviceArray(DeviceArray<T> &&arr);
    DeviceArray &operator=(DeviceArray<T> &&arr);

    ~DeviceArray();

    void reset();
    void deleteDeviceMemory();
    void ensureDeviceAllocation();
    void updateDeviceData();
    void updateHostData();
    
    T *devicePtr() { return m_device; }
    T *hostPtr() { return m_host; }
    const T *hostPtr() const { return m_host; }
    size_t size() const { return static_cast<size_t>(m_size); }

private:
    int m_size;
    T m_initialValue;
    T *m_host;
    T *m_device;
};
