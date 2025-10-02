#pragma once

#include "Utils.hpp"

template<typename T>
struct DeviceVector
{
    DeviceVector(int N, const T &initValue);

    ~DeviceVector();

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

extern template struct DeviceVector<Vec4>;

void printCudaDeviceInfo();
