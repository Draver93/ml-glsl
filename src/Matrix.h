#pragma once

#include <vector>
extern "C" {
#include <glad/glad.h>
}
#include <cstring>
#include <memory>
#include <queue>
#include <mutex>

namespace NNGL {
    class Matrix {
    private:
        bool m_Dirty = true; // Dirty flag for GPU upload
        std::vector<float> flatVec; // Now private: direct access forbidden, use accessors

    public:

        int rows, cols;
        GLuint buffer = 0;

        Matrix() = delete;
        Matrix(int row, int column, float fill = 0.0f);
        Matrix(const std::vector<std::vector<float>>& vec2d);
        Matrix(int r, int c, const float* data);
        Matrix(const Matrix& other);
        ~Matrix();
        Matrix& operator=(const Matrix& other);

        // Element access
        float& operator()(int r, int c); // Sets dirty on write
        float operator()(int r, int c) const;

        // Safe accessors
        float get(int r, int c) const { return (*this)(r, c); }
        void set(int r, int c, float value) { (*this)(r, c) = value; m_Dirty = true; }
        const std::vector<float>& getFlatVec() const { return flatVec; }

        void randomize(float min = -1.0f, float max = 1.0f);
        void add(Matrix& other);
        void add(Matrix& other, float scale);
        void print() const;

        float* raw(); // For GPU upload only
        const float* raw() const;

        size_t byteSize() const;

        void clear(float clear_with = 0.0f);
        void copyFrom(std::shared_ptr<Matrix> srcMat);

        // GPU operations
        void uploadToGPU();
        void allocateBufferGPU();
        void downloadFromGPU();
        
        // Memory management for pooling
        void reserve(size_t size);
        void shrink_to_fit();
        void reset(int newRows, int newCols, float fill = 0.0f);

        // Dirty state
        bool isDirty() const { return m_Dirty; }
    };
}