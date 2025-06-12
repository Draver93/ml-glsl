#pragma once

#include <functional>

namespace NNGL {


    enum ActivationFnType {
        TANH = 0,
        RELU = 1,
        LRELU = 2,
        SIGMOID = 3,
        IDENTITY = 4

    };
    
    struct ActivationFunction {
        std::function<float(float)> func; // should now expect pre-activation
        std::function<float(float)> dfunc; // should now expect pre-activation
        std::function<float(int, int)> weight_initializer;    // weight initializer: (in_size, out_size)
    };

    static std::unordered_map<ActivationFnType, ActivationFunction> activationFunctions = {
        {TANH, {
            [](float x) { return std::tanh(x); },
            [](float z) { float y = std::tanh(z); return 1.0f - y * y; },
            [](int in_size, int out_size) { float range = std::sqrt(6.0f / (in_size + out_size)); return ((float)rand() / RAND_MAX) * 2 * range - range; } //xavier_init
        }},
        {RELU, {
            [](float x) { return std::max(0.0f, x); },
            [](float z) { return z > 0 ? 1.0f : 0.0f; },
            [](int in_size, int /*out_size*/) { float stddev = std::sqrt(2.0f / in_size); return ((float)rand() / RAND_MAX) * 2 * stddev - stddev; } //he_init
        }},
        {LRELU, {
            [](float x) { return x > 0 ? x : 0.01f * x; },
            [](float z) { return z > 0 ? 1.0f : 0.01f; },
            [](int in_size, int /*out_size*/) { float stddev = std::sqrt(2.0f / in_size); return ((float)rand() / RAND_MAX) * 2 * stddev - stddev; } //he_init
        }},
        {SIGMOID, {
            [](float x) { return 1.0f / (1.0f + std::exp(-x)); },
            [](float z) { float y = 1.0f / (1.0f + std::exp(-z)); return y * (1 - y); },
            [](int in_size, int out_size) { float range = std::sqrt(6.0f / (in_size + out_size)); return ((float)rand() / RAND_MAX) * 2 * range - range; } //xavier_init
        }},
        {IDENTITY, {
            [](float x) { return x; },
            [](float) { return 1.0f; },
            [](int in_size, int out_size) { float range = std::sqrt(6.0f / (in_size + out_size)); return ((float)rand() / RAND_MAX) * 2 * range - range; } //xavier_init
        }}
    };
}