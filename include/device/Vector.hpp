#pragma once

#include "device/DeviceMemory.hpp"

#include <vector>
#include <span>

template<typename T>
struct DeviceVector
{
    explicit DeviceVector(std::vector<T> data) : m_host(std::move(data)), m_deviceNeedsUpdate{true} {}
    DeviceVector() = delete;

    void deleteDeviceMemory() {
        m_device.free();
        m_deviceNeedsUpdate = true;
    }

    void ensureDeviceAllocation() {
        if (m_deviceNeedsUpdate) {
            m_device.allocate(sizeof(T) * size());
            updateDeviceData();

            m_deviceNeedsUpdate = false;
        }
    }

    void updateDeviceData() {
        m_device.copyFromHost(m_host.data(), sizeof(T) * size());
    }

    void push_back(T elem) {
        m_host.push_back(std::move(elem));
        m_deviceNeedsUpdate = true;
    }

    T *devicePtr() { return m_device.getPtr<T>(); }
    std::span<T> deviceSpan() { return std::span{devicePtr(), size()}; }

    T *hostPtr() { return m_host.data(); }
    const T *hostPtr() const { return m_host.data(); }
    size_t size() const { return m_host.size(); }

private:
    std::vector<T> m_host;
    DeviceMemoryManager m_device;
    bool m_deviceNeedsUpdate;
};
