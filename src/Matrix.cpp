#include "Matrix.h"

#include <random>
#include <iostream>

namespace NNGL {
    Matrix::Matrix(int r, int c, float fill) : 
        rows(r), cols(c), data(r* c, fill) {
        allocateBufferGPU();
    }

    Matrix::Matrix(const std::vector<std::vector<float>>& vec2d) {
        if (vec2d.empty()) {
            rows = cols = 0;
            return;
        }

        rows = vec2d.size();
        cols = vec2d[0].size();

        // Validate that all rows have the same size
        for (const auto& row : vec2d) {
            if (row.size() != cols) {
                throw std::invalid_argument("All rows must have the same number of columns");
            }
        }

        // Reserve space and copy data row by row
        data.reserve(rows * cols);
        for (const auto& row : vec2d) {
            data.insert(data.end(), row.begin(), row.end());
        }
        allocateBufferGPU();
    }

    // Access element (row, col)
    float& Matrix::operator()(int r, int c) {
        if (r >= rows || c >= cols) throw std::out_of_range("Matrix index out of bounds");
        return data[r * cols + c];
    }

    float Matrix::operator()(int r, int c) const {
        if (r >= rows || c >= cols) throw std::out_of_range("Matrix index out of bounds");
        return data[r * cols + c];
    }

    // Fill with random (for example init weights)
    void Matrix::randomize(float min, float max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);
        for (auto& val : data)
            val = dist(gen);
    }

    // Multiply with another matrix (this * B)
    Matrix Matrix::multiply(const Matrix& B) const {
        if (cols != B.rows) throw std::invalid_argument("Matrix multiplication shape mismatch");
        Matrix result(rows, B.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < B.cols; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < cols; ++k) {
                    sum += (*this)(i, k) * B(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    // Print for debugging
    void Matrix::print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                std::cout << (*this)(i, j) << ' ';
            std::cout << '\n';
        }
    }

    // Return raw pointer (useful for GPU upload)
    float* Matrix::raw() { return data.data(); }
    const float* Matrix::raw() const { return data.data(); }

    // Total size in bytes
    size_t Matrix::byteSize() const { return data.size() * sizeof(float); }

    void Matrix::allocateBufferGPU() {
        if (buffer != 0) glDeleteBuffers(1, &buffer);
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, byteSize(), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    void Matrix::uploadToGPU() {
        if (buffer == 0) allocateBufferGPU();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, byteSize(), data.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    void Matrix::downloadFromGPU() {
        if (buffer == 0) return;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, byteSize(), data.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

}