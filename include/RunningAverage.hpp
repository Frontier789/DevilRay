#pragma once

#include <deque>

struct RunningAverage
{
    RunningAverage(int sampleCount);

    void add(double value);
    double mean() const;
    void reset();

private:
    std::deque<double> m_samples;
    int m_sampleCount;
};
