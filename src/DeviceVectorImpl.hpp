template<typename T>
DeviceVector<T>::DeviceVector(int N, const T &initValue)
    : m_size(N)
    , m_initialValue(initValue)
    , m_host(new T[N])
    , m_device(nullptr)
{
    
}

template<typename T>
DeviceVector<T>::~DeviceVector()
{
    delete[] m_host;
    deleteDeviceMemory();
}

template<typename T>
void DeviceVector<T>::reset()
{
    std::fill_n(m_host, m_size, m_initialValue);

    if (m_device)
    {
        cudaMemcpy(m_device, m_host, sizeof(T)*m_size, cudaMemcpyHostToDevice);
    }
}

template<typename T>
void DeviceVector<T>::ensureDeviceAllocation()
{
    if (!m_device) {
        cudaMalloc(&m_device, sizeof(T)*m_size);
        reset();
    }
}

template<typename T>
void DeviceVector<T>::updateHostData()
{
    if (!m_device) return;

    cudaMemcpy(m_host, m_device, sizeof(T)*m_size, cudaMemcpyDeviceToHost);
}

template<typename T>
void DeviceVector<T>::deleteDeviceMemory()
{
    cudaFree(m_device);
    m_device = nullptr;
}