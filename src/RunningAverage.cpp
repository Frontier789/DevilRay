#include "RunningAverage.hpp"

#include <numeric>

RunningAverage::RunningAverage(int sampleCount) : m_sampleCount(sampleCount) {}

void RunningAverage::add(double value) {
    m_samples.push_back(value);
    if (m_samples.size() > m_sampleCount) {
        m_samples.pop_front();
    }
}

double RunningAverage::mean() const {
    if (m_samples.empty()) return 0;

    return std::accumulate(m_samples.begin(), m_samples.end(), 0.0) / m_samples.size();
}

void RunningAverage::reset() { m_samples.clear(); }
