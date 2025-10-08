#pragma once

#include "Utils.hpp"

template<typename T>
struct DeviceArray
{
    DeviceArray(int N, const T &initValue);

    ~DeviceArray();

    void reset();
    void deleteDeviceMemory();
    void ensureDeviceAllocation();
    void updateHostData();
    
    T *devicePtr() { return m_device; }
    T *hostPtr() { return m_host; }
    int size() const { return m_size; }

private:
    int m_size;
    T m_initialValue;
    T *m_host;
    T *m_device;
};

extern template struct DeviceArray<Vec4>;
