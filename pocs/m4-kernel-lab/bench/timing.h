#pragma once

// Tiny portable timing helpers for the kernel lab.
// On macOS we use mach_absolute_time; on Linux clock_gettime(CLOCK_MONOTONIC).

#include <stdint.h>

#if defined(__APPLE__)
#  include <mach/mach_time.h>
static inline double timing_now_ns(void) {
    static mach_timebase_info_data_t info = { 0, 0 };
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    const uint64_t t = mach_absolute_time();
    return (double) t * (double) info.numer / (double) info.denom;
}
#else
#  include <time.h>
static inline double timing_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e9 + (double) ts.tv_nsec;
}
#endif
