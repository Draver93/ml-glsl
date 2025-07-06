#include "Matrix.h"

#include <random>
#include <iostream>

namespace NNGL {
    Matrix::Matrix(int r, int c, float fill) : 
        rows(r), cols(c), flatVec(r * c, fill) {
        allocateBufferGPU();
    }

    Matrix::Matrix(int r, int c, const float* data) : rows(r), cols(c) {
        if (r <= 0 || c <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
        if (data == nullptr) {
            throw std::invalid_argument("Data pointer cannot be null");
        }

        // Reserve and resize the flat vector
        flatVec.reserve(rows * cols);
        flatVec.resize(rows * cols);

        // Direct copy since both input and storage are column-major
        std::copy(data, data + (rows * cols), flatVec.begin());
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

        // Reserve space
        flatVec.reserve(rows * cols);
        flatVec.resize(rows * cols);

        // Copy data in COLUMN-MAJOR order
        // For each column, copy all row values for that column
        for (int c = 0; c < cols; ++c) {        // For each column
            for (int r = 0; r < rows; ++r) {    // For each row in that column
                flatVec[c * rows + r] = vec2d[r][c];  // Column-major indexing
            }
        }

        allocateBufferGPU();
    }

    Matrix::~Matrix() {
        if (buffer != 0) {
            std::cout << "[GPU BUFFER] Deleting buffer " << buffer 
                      << " for matrix " << rows << "x" << cols << std::endl;
            glDeleteBuffers(1, &buffer);
            buffer = 0;
        }
    }

    Matrix::Matrix(const Matrix& other)
        : rows(other.rows), cols(other.cols), flatVec(other.flatVec), buffer(0) {
    }

    Matrix& Matrix::operator=(const Matrix& other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            flatVec = other.flatVec;
        }
        return *this;
    }

    // Column-major indexing: data stored as [col0_all_rows, col1_all_rows, ...]
    float& Matrix::operator()(int r, int c) {
        if (r >= rows || c >= cols) throw std::out_of_range("Matrix index out of bounds");
        return flatVec[c * rows + r];  // Column-major: column_index * num_rows + row_index
    }

    float Matrix::operator()(int r, int c) const {
        if (r >= rows || c >= cols) throw std::out_of_range("Matrix index out of bounds");
        return flatVec[c * rows + r];
    }

    // Fill with random (for example init weights)
    void Matrix::randomize(float min, float max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);
        for (auto& val : flatVec)
            val = dist(gen);
    }

    // Print for debugging
    void Matrix::print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                std::cout << (*this)(i, j) << ' ';
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    void Matrix::add(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) {
            throw std::runtime_error("Matrix dimensions must match for addition");
        }

        for (size_t i = 0; i < flatVec.size(); ++i) {
            flatVec[i] += other.flatVec[i];
        }
    }

    // Return raw pointer (useful for GPU upload)
    float* Matrix::raw() { return flatVec.data(); }
    const float* Matrix::raw() const { return flatVec.data(); }

    // Total size in bytes
    size_t Matrix::byteSize() const { return flatVec.size() * sizeof(float); }

    void Matrix::allocateBufferGPU() {
        if (buffer != 0) {
            std::cout << "[GPU BUFFER] Deleting existing buffer " << buffer << std::endl;
            glDeleteBuffers(1, &buffer);
        }
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, byteSize(), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        
        // Log GPU buffer allocation
        std::cout << "[GPU BUFFER] Allocated new buffer " << buffer 
                  << " for matrix " << rows << "x" << cols 
                  << " (" << byteSize() << " bytes)" << std::endl;
    }

    void Matrix::uploadToGPU() {
        if (buffer == 0) allocateBufferGPU();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, byteSize(), flatVec.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        
        // Log GPU upload information
        std::cout << "[GPU UPLOAD] Matrix " << rows << "x" << cols 
                  << " (" << byteSize() << " bytes) uploaded to GPU buffer " << buffer << std::endl;
    }

    void Matrix::downloadFromGPU() {
        if (buffer == 0) {
            std::cout << "[GPU DOWNLOAD] Skipped - no GPU buffer allocated" << std::endl;
            return;
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, byteSize(), flatVec.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        
        // Log GPU download information
        std::cout << "[GPU DOWNLOAD] Matrix " << rows << "x" << cols 
                  << " (" << byteSize() << " bytes) downloaded from GPU buffer " << buffer << std::endl;
    }

    void Matrix::copyFrom(std::shared_ptr<Matrix> srcMat) {
        if (!srcMat) {
            throw std::invalid_argument("Source matrix is null");
        }

        // Check dimensions match
        if (this->rows != srcMat->rows || this->cols != srcMat->cols) {
            throw std::invalid_argument("Matrix dimensions must match for copyFrom operation");
        }

        // Copy the data from source matrix's flatVec to this matrix's flatVec
        std::memcpy(this->flatVec.data(), srcMat->flatVec.data(), this->byteSize());
    }

    // Memory management for pooling
    void Matrix::reserve(size_t size) {
        flatVec.reserve(size);
    }

    void Matrix::shrink_to_fit() {
        flatVec.shrink_to_fit();
    }

    void Matrix::reset(int newRows, int newCols, float fill) {
        rows = newRows;
        cols = newCols;
        
        // Reuse existing memory if possible
        size_t newSize = rows * cols;
        if (flatVec.capacity() >= newSize) {
            flatVec.resize(newSize, fill);
        } else {
            flatVec.clear();
            flatVec.resize(newSize, fill);
        }
        
        // Reset GPU buffer if dimensions changed
        if (buffer != 0) {
            std::cout << "[GPU BUFFER] Resetting buffer " << buffer 
                      << " due to dimension change from " << rows << "x" << cols 
                      << " to " << newRows << "x" << newCols << std::endl;
            glDeleteBuffers(1, &buffer);
            buffer = 0;
        }
    }
}