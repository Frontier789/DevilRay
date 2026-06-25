#pragma once

#include "device/DeviceMemory.hpp"

#include <algorithm>
#include <span>
#include <vector>

template<typename T>
struct DeviceArray
{
    DeviceArray() : DeviceArray(0, T{}) {}
    DeviceArray(int N, const T &initValue)
        : m_initialValue(initValue)
        , m_host(static_cast<size_t>(N), initValue)
    {
    }

    void reset() {
        std::fill(m_host.begin(), m_host.end(), m_initialValue);
        updateDeviceData();
    }

    void deleteDeviceMemory() { m_device.free(); }

    void ensureDeviceAllocation() {
        if (m_device.empty()) {
            m_device.allocate(sizeof(T) * size());
            updateDeviceData();
        }
    }

    void updateDeviceData() {
        if (m_device.empty()) return;
        m_device.copyFromHost(m_host.data(), sizeof(T) * size());
    }

    void updateHostData() {
        if (m_device.empty()) return;
        m_device.copyToHost(m_host.data(), sizeof(T) * size());
    }

    T *devicePtr() { return m_device.getPtr<T>(); }
    T *hostPtr() { return m_host.data(); }
    const T *hostPtr() const { return m_host.data(); }
    std::span<T> hostSpan() { return std::span{m_host.data(), size()}; }
    size_t size() const { return m_host.size(); }

private:
    T m_initialValue;
    std::vector<T> m_host;
    DeviceMemoryManager m_device;
};
