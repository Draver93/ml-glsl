#pragma once

#include <vector>
extern "C" {
#include <glad/glad.h>
}
#include <memory>

namespace NNGL {
    struct Matrix {
        std::vector<float> flatVec;
        int rows, cols;

        GLuint buffer = 0;

        Matrix() = delete;
        Matrix(int row, int column, float fill = 0.0f);
        Matrix(const std::vector<std::vector<float>>& vec2d);
        Matrix(int r, int c, const float* data);

        Matrix(const Matrix& other);
        ~Matrix();

        Matrix& operator=(const Matrix& other);
        float& operator()(int r, int c);
        float operator()(int r, int c) const;

        void randomize(float min = -1.0f, float max = 1.0f);
        Matrix multiply(const Matrix& B) const;
        void add(const Matrix& other);
        void print() const;

        float* raw();
        const float* raw() const;

        size_t byteSize() const;

        void clear(float clear_with = 0.0f) { std::memset(flatVec.data(), clear_with, flatVec.size() * sizeof(float)); }

        void copyFrom(std::shared_ptr<Matrix> srcMat);

        void uploadToGPU();
        void allocateBufferGPU();
        void downloadFromGPU();
    };
}