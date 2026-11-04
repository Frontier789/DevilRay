#pragma once

#include "device/DeviceMemory.hpp"

#include <algorithm>
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <vector>

template<typename T>
struct DefaultInitAllocator
{
    using value_type = T;

    DefaultInitAllocator() = default;

    template<typename U>
    DefaultInitAllocator(const DefaultInitAllocator<U> &) noexcept {}

    T *allocate(size_t n) { return std::allocator<T>{}.allocate(n); }
    void deallocate(T *p, size_t n) noexcept { std::allocator<T>{}.deallocate(p, n); }

    template<typename U>
    void construct(U *p) { ::new (static_cast<void *>(p)) U; }

    template<typename U, typename... Args>
    void construct(U *p, Args &&...args) { ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...); }

    template<typename U>
    bool operator==(const DefaultInitAllocator<U> &) const noexcept { return true; }
};

template<typename T>
struct DeviceArray
{
    DeviceArray() : DeviceArray(0, T{}) {}
    DeviceArray(int N, const T &initValue)
        : m_initialValue(initValue)
        , m_host(static_cast<size_t>(N), initValue)
    {
    }

    static DeviceArray uninitialized(int N)
    {
        DeviceArray array;
        array.m_host.resize(static_cast<size_t>(N));
        return array;
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
    std::span<const T> hostSpan() const { return std::span<const T>{m_host.data(), size()}; }
    std::span<T> deviceSpan() { return std::span{devicePtr(), size()}; }
    size_t size() const { return m_host.size(); }

private:
    T m_initialValue;
    std::vector<T, DefaultInitAllocator<T>> m_host;
    DeviceMemoryManager m_device;
};
