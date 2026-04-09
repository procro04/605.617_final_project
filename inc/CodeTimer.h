#pragma once

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>


struct TimingRecord {
    std::string label;
    double      elapsed_ms;
};

// Scoped timer for even easier use
class ScopedTimer {
public:
    ScopedTimer(const std::string& label, std::vector<TimingRecord>& log)
        : label_(label), log_(log),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        log_.push_back({label_, ms});
    }

private:
    std::string label_;
    std::vector<TimingRecord>& log_;
    std::chrono::high_resolution_clock::time_point start_;
};

// Code timer for more manual use
class CodeTimer
{
public:
    CodeTimer()
        : m_running(false)
        , m_startTime{}
        , m_endTime{}
    {}

    void startTiming()
    {
        if (m_running)
            throw std::runtime_error("TimingClass: Timer is already running. Call stopTiming() first.");

        m_running   = true;
        m_startTime = std::chrono::high_resolution_clock::now();
    }

    void stopTiming()
    {
        if (!m_running)
            throw std::runtime_error("TimingClass: Timer is not running. Call startTiming() first.");

        m_endTime = std::chrono::high_resolution_clock::now();
        m_running = false;
    }

    void timingResults() const
    {
        if (m_running)
            throw std::runtime_error("TimingClass: Timer is still running. Call stopTiming() first.");

        // Duration as floating-point seconds — fractional seconds are preserved
        const std::chrono::duration<double> elapsed = m_endTime - m_startTime;

        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    }

    // Convenience method if you just want the raw value back
    double elapsedSeconds() const
    {
        if (m_running)
            throw std::runtime_error("TimingClass: Timer is still running. Call stopTiming() first.");

        const std::chrono::duration<double> elapsed = m_endTime - m_startTime;
        return elapsed.count();
    }

private:
    bool                                                     m_running;
    std::chrono::high_resolution_clock::time_point          m_startTime;
    std::chrono::high_resolution_clock::time_point          m_endTime;
};
