#pragma once

/*
Cmake macro:
    - DEBUG
*/

#if DEBUG
    #ifndef DEBUG_SPDLOG
        #define DEBUG_SPDLOG 1
    #endif
#endif