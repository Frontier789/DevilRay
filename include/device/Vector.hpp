#pragma once

#include <vector>

struct DeviceMemoryManager
{
    DeviceMemoryManager();
    DeviceMemoryManager(const DeviceMemoryManager &) = delete;
    DeviceMemoryManager(DeviceMemoryManager &&);
    ~DeviceMemoryManager();

    DeviceMemoryManager &operator=(DeviceMemoryManager &&other);

    template<typename T>
    T *getPtr() {return static_cast<T*>(m_devicePtr);}

    void allocate(size_t bytes);
    void free();
    void memcpy(const void *src, size_t bytes);
    bool empty() const;

private:
    void *m_devicePtr;
};

template<typename T>
struct DeviceVector
{
    explicit DeviceVector(std::vector<T> data) : m_host(std::move(data)), m_deviceNeedsUpdate{true} {}
    DeviceVector() : m_deviceNeedsUpdate{true} {}

    void deleteDeviceMemory() {
        m_device.free();
        m_deviceNeedsUpdate = true;
    }

    void ensureDeviceAllocation() {
        if (m_deviceNeedsUpdate) {
            m_device.allocate(sizeof(T) * size());
            updateDeviceData();
        }
    }

    void updateDeviceData() {
        m_device.memcpy(m_host.data(), sizeof(T) * size());
    }

    void push_back(T elem) {
        m_host.push_back(std::move(elem));
        m_deviceNeedsUpdate = true;
    }
    
    T *devicePtr() { return m_device.getPtr<T>(); }
    T *hostPtr() { return m_host.data(); }
    const T *hostPtr() const { return m_host.data(); }
    size_t size() const { return m_host.size(); }

private:
    std::vector<T> m_host;
    DeviceMemoryManager m_device;
    bool m_deviceNeedsUpdate;
};

