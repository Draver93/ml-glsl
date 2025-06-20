#pragma once

#include <vector>
extern "C" {
#include <glad/glad.h>
}

namespace NNGL {
    struct Matrix {
        std::vector<float> data;
        int rows, cols;

        GLuint buffer = 0;

        Matrix() = delete;
        Matrix(int r, int c, float fill = 0.0f);
        Matrix(const std::vector<std::vector<float>>& vec2d);

        float& operator()(int r, int c);
        float operator()(int r, int c) const;

        void randomize(float min = -1.0f, float max = 1.0f);
        Matrix multiply(const Matrix& B) const;
        void print() const;

        float* raw();
        const float* raw() const;

        size_t byteSize() const;

        void uploadToGPU();
        void allocateBufferGPU();
        void downloadFromGPU();
    };
}