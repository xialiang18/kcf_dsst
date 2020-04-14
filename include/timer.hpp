// Copyright [2019] <liangding@sensetime.com>
#pragma once

// #ifndef PERFORMANCE
// #define Timer(name)
// #else
// #define Timer(name) TimeLogger timer(name)

#define Timer_Begin(name) TimeLogger *timer##name = new TimeLogger(#name)
#define Timer_End(name) timer##name->~TimeLogger();

#include <map>
#include <vector>
#include <chrono>
#include <iostream>

class TimeLogger
{
    public:
    TimeLogger(const std::string &name) : name_(name), start_(std::chrono::system_clock::now())
    {
    }
    ~TimeLogger()
    {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - start_);
        durations_[name_].push_back(duration);
        printf("> Duration [%s]: %.5f ms\n", name_.c_str(),
                     duration.count() / 1000.0);
    }

    static void Print(const std::string &name)
    {
        // ignore first log duration if vector size is greater than 1
        int index = durations_[name].size() > 1 ? -1 : 0;
        int64_t total_milliseconds = 0;

        for (const auto &duration : durations_[name]) {
            ++index;
            if (index) {
                total_milliseconds += duration.count() / 1000.0;
            }
        }
        double time_in_milliseconds = total_milliseconds / index;
        printf("> Benchmark [%s]: %.5f ms\n", name.c_str(),
                     time_in_milliseconds);
    }

    static void PrintAll()
    {
        for (auto it : durations_) {
            Print(it.first);
        }
    }

    private:
    std::string name_;
    std::chrono::time_point<std::chrono::system_clock> start_;

    static std::map<std::string, std::vector<std::chrono::microseconds>> durations_;
};

//#endif