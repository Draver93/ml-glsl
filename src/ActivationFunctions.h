#pragma once

#include <functional>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "Matrix.h"

namespace NNGL {
    // Helper to print a 5x5 slice of a matrix
    static void printMatrixSlice(const std::string& name, const std::shared_ptr<NNGL::Matrix>& mat) {
        return;
        if (!mat) { std::cout << name << ": nullptr" << std::endl; return; }
        std::cout << "[DEBUG] " << name << " shape=[" << mat->rows << "," << mat->cols << "]\n";
        int max_rows = std::min(5, mat->rows);
        int max_cols = std::min(5, mat->cols);
        // Print first 5 rows
        for (int r = 0; r < max_rows; ++r) {
            std::cout << "  ";
            for (int c = 0; c < max_cols; ++c) {
                std::cout << std::setw(8) << std::setprecision(4) << (*mat)(r, c) << " ";
            }
            if (mat->cols > 5) std::cout << "... ";
            // Print last 5 columns if matrix is wide
            if (mat->cols > 10) {
                for (int c = mat->cols - 5; c < mat->cols; ++c) {
                    std::cout << std::setw(8) << std::setprecision(4) << (*mat)(r, c) << " ";
                }
            }
            std::cout << std::endl;
        }
        if (mat->rows > 5) std::cout << "  ..." << std::endl;
        // Print last 5 rows if matrix is tall
        if (mat->rows > 10) {
            for (int r = mat->rows - 5; r < mat->rows; ++r) {
                std::cout << "  ";
                for (int c = 0; c < max_cols; ++c) {
                    std::cout << std::setw(8) << std::setprecision(4) << (*mat)(r, c) << " ";
                }
                if (mat->cols > 5) std::cout << "... ";
                if (mat->cols > 10) {
                    for (int c = mat->cols - 5; c < mat->cols; ++c) {
                        std::cout << std::setw(8) << std::setprecision(4) << (*mat)(r, c) << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
    }

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