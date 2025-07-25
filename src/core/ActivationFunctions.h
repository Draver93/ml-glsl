#pragma once

#include <functional>
#include <algorithm>
#include <cmath>
#include <vector>

namespace NNGL {
    static std::vector<float> softmax(const std::vector<float>& input) {
        std::vector<float> output(input.size());

        float maxInput = *std::max_element(input.begin(), input.end());

        float sumExp = 0.0f;
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i] - maxInput);
            sumExp += output[i];
        }

        for (size_t i = 0; i < output.size(); ++i) {
            output[i] /= sumExp;
        }

        return output;
    }

    enum ActivationFnType {
        TANH = 0,
        RELU = 1,
        LRELU = 2,
        SIGMOID = 3,
        IDENTITY = 4

    };
    
    struct ActivationFunction {
        std::function<float(float)> func; 
        std::function<float(float)> dfunc;
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