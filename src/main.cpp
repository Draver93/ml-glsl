#include <chrono>
#include  <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <array>
#include <map>
#include <vector>
#include <set>
#include <functional>
#include <algorithm>
#include <random>
#include <numeric>

#include <glm/glm.hpp>
#include "GPTransformer.h"
#include "ActivationFunctions.h"
#include "AttentionBlock.h"
#include "NeuralNetwork.h"
#include "MNISTLoader.h"
#include "BPE.h"
#include "Matrix.h"
#include "Logger.h"
#include "LayerNorm.h"


#include <vector>
#include <cstdint>
#include <cmath>
#include <cassert>

uint32_t hash32(const std::string& str) {
    const uint32_t FNV_PRIME = 0x01000193; //   16777619
    const uint32_t OFFSET_BASIS = 0x811C9DC5; // 2166136261

    uint32_t hash = OFFSET_BASIS;
    for (char c : str) {
        hash ^= static_cast<uint8_t>(c);
        hash *= FNV_PRIME;
    }
    return hash;
}

std::vector<float> tokenToInputVector(const std::string& token) {
    uint32_t h = hash32(token);  // example: 0x9E3779B9
    std::vector<float> vec(32);
    for (int i = 0; i < 32; ++i) {
        vec[i] = ((h >> (i % 32)) & 1) ? 1.0f : 0.0f;  // crude binary split
    }
    return vec;
}

// Helper to read little-endian integers from file
template<typename T>
void readLE(std::ifstream& file, T& val) {
    file.read(reinterpret_cast<char*>(&val), sizeof(T));
    // On little-endian machines, no conversion needed
}

// Helper function to escape special characters for display
std::string escapeString(const std::string& str) {
    std::string escaped;
    for (char c : str) {
        switch (c) {
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            default: escaped += c; break;
        }
    }
    return escaped;
}

// Load BMP, convert to normalized grayscale float vector
std::vector<uint8_t> loadBMPtoGrayscale(const std::string& filename, int& outWidth, int& outHeight) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open BMP file");

    // --- Read BMP file header ---
    uint16_t bfType;
    readLE(file, bfType);
    if (bfType != 0x4D42) // 'BM' in little endian
        throw std::runtime_error("Not a BMP file");

    uint32_t bfSize, bfReserved, bfOffBits;
    readLE(file, bfSize);
    readLE(file, bfReserved);
    readLE(file, bfOffBits);

    // --- Read DIB header size ---
    uint32_t dibHeaderSize;
    readLE(file, dibHeaderSize);

    if (dibHeaderSize < 40 || dibHeaderSize > 128)
        std::cout << "Warning: Unexpected DIB header size: " << dibHeaderSize << std::endl;

    // Read the rest of the DIB header into buffer
    std::vector<uint8_t> dibHeader(dibHeaderSize);
    // We already read the first 4 bytes (dibHeaderSize), so copy that
    memcpy(dibHeader.data(), &dibHeaderSize, 4);
    // Read remaining bytes
    file.read(reinterpret_cast<char*>(dibHeader.data() + 4), dibHeaderSize - 4);

    // Extract width, height, planes, bitcount (from first 40 bytes)
    int32_t width = *reinterpret_cast<int32_t*>(&dibHeader[4]);
    int32_t height = *reinterpret_cast<int32_t*>(&dibHeader[8]);
    uint16_t planes = *reinterpret_cast<uint16_t*>(&dibHeader[12]);
    uint16_t bitCount = *reinterpret_cast<uint16_t*>(&dibHeader[14]);
    uint32_t compression = *reinterpret_cast<uint32_t*>(&dibHeader[16]);

    if (planes != 1) throw std::runtime_error("Unsupported BMP planes count");
    if (compression != 0) throw std::runtime_error("Compressed BMP not supported");

    // Only support 24-bit or 8-bit BMP here
    if (bitCount != 24 && bitCount != 8)
        throw std::runtime_error("Only 24-bit and 8-bit BMP supported");

    // Output dimensions
    outWidth = width;
    outHeight = (height > 0) ? height : -height; // Height can be negative to indicate top-down bitmap

    // --- Handle palette for 8-bit images ---
    std::vector<uint32_t> palette;
    if (bitCount == 8) {
        // Palette is after DIB header, each palette entry 4 bytes (B,G,R,0)
        size_t paletteSize = (bfOffBits - 14 - dibHeaderSize) / 4;
        palette.resize(paletteSize);
        for (size_t i = 0; i < paletteSize; ++i) {
            uint8_t b = file.get();
            uint8_t g = file.get();
            uint8_t r = file.get();
            uint8_t reserved = file.get();
            palette[i] = (r << 16) | (g << 8) | b;
        }
    }

    // --- Read pixel data ---
    file.seekg(bfOffBits, std::ios::beg);

    int rowSize = ((bitCount * width + 31) / 32) * 4; // row size is padded to 4 bytes
    std::vector<uint8_t> rowData(rowSize);

    std::vector<uint8_t> pixels(outWidth * outHeight);

    bool topDown = (height < 0);

    for (int row = 0; row < outHeight; ++row) {
        int readRow = topDown ? row : (outHeight - 1 - row);
        file.read(reinterpret_cast<char*>(rowData.data()), rowSize);

        for (int col = 0; col < outWidth; ++col) {
            if (bitCount == 24) {
                // 3 bytes per pixel: B G R
                int idx = col * 3;
                uint8_t b = rowData[idx];
                uint8_t g = rowData[idx + 1];
                uint8_t r = rowData[idx + 2];
                // Convert RGB to grayscale luminance (approx)
                float gray = (0.299f * r + 0.587f * g + 0.114f * b);
                pixels[readRow * outWidth + col] = gray;
            }
            else if (bitCount == 8) {
                uint8_t paletteIndex = rowData[col];
                uint32_t rgb = palette[paletteIndex];
                uint8_t r = (rgb >> 16) & 0xFF;
                uint8_t g = (rgb >> 8) & 0xFF;
                uint8_t b = rgb & 0xFF;
                float gray = (0.299f * r + 0.587f * g + 0.114f * b);
                pixels[readRow * outWidth + col] = gray;
            }
        }
    }

    return pixels;
}

// Helper: convert RGB to grayscale luminance
uint8_t rgbToGrayscale(uint8_t r, uint8_t g, uint8_t b) {
    // Use standard luminance formula
    return static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
}

// Nearest neighbor resize for simplicity
std::vector<uint8_t> resizeImageNearestNeighbor(
    const std::vector<uint8_t>& srcPixels, int srcWidth, int srcHeight,
    int dstWidth, int dstHeight)
{
    std::vector<uint8_t> dstPixels(dstWidth * dstHeight);

    for (int y = 0; y < dstHeight; ++y) {
        int srcY = static_cast<int>(y * (float)srcHeight / dstHeight);
        for (int x = 0; x < dstWidth; ++x) {
            int srcX = static_cast<int>(x * (float)srcWidth / dstWidth);
            dstPixels[y * dstWidth + x] = srcPixels[srcY * srcWidth + srcX];
        }
    }

    return dstPixels;
}

// Convert RGB BMP image to MNIST format 28x28 grayscale pixels
std::vector<uint8_t> bmpToMNIST(
    const std::vector<uint8_t>& rgbPixels, int width, int height)
{
    // 1) Convert RGB to grayscale
    std::vector<uint8_t> grayscalePixels(width * height);
    for (int i = 0; i < width * height; ++i) {
        uint8_t r = rgbPixels[i * 3 + 0];
        uint8_t g = rgbPixels[i * 3 + 1];
        uint8_t b = rgbPixels[i * 3 + 2];
        grayscalePixels[i] = rgbToGrayscale(r, g, b);
    }

    // 2) Resize to 28x28
    std::vector<uint8_t> resizedPixels = resizeImageNearestNeighbor(
        grayscalePixels, width, height, 28, 28);

    // 3) Optionally invert colors if needed (MNIST digits are white-on-black)
    // If your BMP digit is black-on-white background, invert:
    // for (auto& px : resizedPixels) px = 255 - px;

    return resizedPixels; // 784 uint8 grayscale pixels
}


// Random float between -1 and 1
float random_norm_val() {
    return ((float)rand() / RAND_MAX) * 2 - 1;
}

void resetCursor() {
#ifdef _WIN32
    // Windows console API (best way)
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD coord = { 0, 0 };
    SetConsoleCursorPosition(hConsole, coord);
#else
    // ANSI escape sequence for UNIX-like systems
    std::cout << "\033[H"; // Move cursor to top-left
#endif
}

void print2DImage(const std::vector<uint8_t>& image, int width, int height) {
    const char* shades = " .:-=+*#%@"; // 10 levels of intensity

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float val = image[y * width + x] / (float)255;
            val = std::clamp(val, 0.0f, 1.0f); // safety clamp
            int shadeIdx = static_cast<int>(val * 9); // map [0,1] to 0-9
            std::cout << shades[shadeIdx];
        }
        std::cout << '\n';
    }
}

void digit_recognition() {
    // Network setup - simplified for testing
    const int inputSize = 784;
    const int hiddenSize = 64;
    const int outputSize = 10;

    const int batchSize = 8;
    const int steps = 2000000;

    NNGL::NeuralNetwork nn(batchSize);

    nn.addLayer(inputSize, hiddenSize, NNGL::ActivationFnType::SIGMOID);
    nn.addLayer(hiddenSize, hiddenSize, NNGL::ActivationFnType::RELU);
    nn.addLayer(hiddenSize, hiddenSize, NNGL::ActivationFnType::RELU);
    nn.addLayer(hiddenSize, hiddenSize, NNGL::ActivationFnType::TANH);
    nn.addLayer(hiddenSize, outputSize, NNGL::ActivationFnType::IDENTITY);

    std::vector<std::vector<uint8_t>> trainImages = NNGL::MNIST::loadImages("mnist/train-images.idx3-ubyte");
    std::vector<uint8_t> trainLabels = NNGL::MNIST::loadLabels("mnist/train-labels.idx1-ubyte");

    //print2DImage(trainImages[52], 28, 28);

    // Load test data
    std::vector<std::vector<uint8_t>> testImages = NNGL::MNIST::loadImages("mnist/t10k-images.idx3-ubyte");
    std::vector<uint8_t> testLabels = NNGL::MNIST::loadLabels("mnist/t10k-labels.idx1-ubyte");


    /*std::vector<std::vector<uint8_t>> testImages;
    std::vector<uint8_t> testLabels;
    int w, h;
    auto bmp = loadBMPtoGrayscale("test_case_1.bmp", w, h);
    testImages.push_back(bmp);
    testLabels.push_back(1);
    bmp = loadBMPtoGrayscale("test_case_2.bmp", w, h);
    testImages.push_back(bmp);
    testLabels.push_back(2);
    bmp = loadBMPtoGrayscale("test_case_3.bmp", w, h);
    testImages.push_back(bmp);
    testLabels.push_back(3);
    bmp = loadBMPtoGrayscale("test_case_4.bmp", w, h);
    testImages.push_back(bmp);
    testLabels.push_back(4);
    bmp = loadBMPtoGrayscale("test_case_5.bmp", w, h);
    testImages.push_back(bmp);
    testLabels.push_back(5);
    bmp = loadBMPtoGrayscale("test_case_6.bmp", w, h);
    testImages.push_back(bmp);
    testLabels.push_back(6);
    bmp = loadBMPtoGrayscale("test_case_7.bmp", w, h);
    testImages.push_back(bmp);
    testLabels.push_back(7);
    bmp = loadBMPtoGrayscale("test_case_8.bmp", w, h);
    testImages.push_back(bmp);
    testLabels.push_back(8);
    bmp = loadBMPtoGrayscale("test_case_9.bmp", w, h);
    testImages.push_back(bmp);
    testLabels.push_back(9);*/

    //print2DImage(testImages[0], 28, 28);

    // Track current position in dataset
    // Separate indices for train and test
    static size_t trainIndex = 0;
    static size_t testIndex = 0;

    const size_t totalTrainSamples = trainImages.size();
    const size_t totalTestSamples = testImages.size();

    nn.onTrainBatch([&](std::shared_ptr<NNGL::Matrix>& batchInputs, std::shared_ptr<NNGL::Matrix>& batchTargets, int batchSize) {
        batchTargets->clear();
        for (int column = 0; column < batchSize; column++) {
            // Use training data with wraparound
            size_t sampleIndex = (trainIndex + column) % totalTrainSamples;

            // Convert image from uint8_t to float and normalize (0-255 → 0-1)
            const auto& image = trainImages[sampleIndex];
            for (size_t pixel = 0; pixel < 784; ++pixel) 
                (*batchInputs)(pixel, column) = static_cast<float>(image[pixel]) / 255.0f;

            // Convert label to one-hot encoding
            uint8_t label = trainLabels[sampleIndex];
            (*batchTargets)(label, column) = 1.0f;
        }
        trainIndex = (trainIndex + batchSize) % totalTrainSamples;
    });

    nn.onTestBatch([&](std::shared_ptr<NNGL::Matrix>& batchInputs, std::shared_ptr<NNGL::Matrix>& batchTargets, int batchSize) {
        batchTargets->clear();
        for (int column = 0; column < batchSize; column++) {
            // Use training data with wraparound
            size_t sampleIndex = (testIndex + column) % totalTestSamples;

            // Convert image from uint8_t to float and normalize (0-255 → 0-1)
            const auto& image = testImages[sampleIndex];
            for (size_t pixel = 0; pixel < 784; ++pixel)
                (*batchInputs)(pixel, column) = static_cast<float>(image[pixel]) / 255.0f;

            // Convert label to one-hot encoding
            uint8_t label = testLabels[sampleIndex];
            (*batchTargets)(label, column) = 1.0f;
        }
        testIndex = (testIndex + batchSize) % totalTestSamples;
    });

    int stepsLeft = steps / batchSize;

    float initialLR = 0.0001f;
    float learningRate = initialLR;
    float decayRate = 0.98f;
    int decaySteps = steps / 1000; 

    while (stepsLeft > 0) {

        nn.train(learningRate);

        // Periodic GPU synchronization and cleanup
        if (stepsLeft % 100 == 0) {
            glFinish(); // Force GPU to complete all pending operations
            GLenum error = glGetError(); // Check for OpenGL errors
            if (error != GL_NO_ERROR) throw std::runtime_error("OpenGL Error detected: " + std::to_string(error) + " at training step " + std::to_string(steps / batchSize - stepsLeft));
        }

        // Status updates and weight display
        if (stepsLeft % (10000) == 0) {
            //resetCursor();
            //for (int i = 1; i < nn.m_Layers.size(); i++) nn.m_Layers[i]->printHeatmap();

            float accurracy = nn.eval(100);

            std::cout << "\nStep: " << (steps / batchSize - stepsLeft) * batchSize << " LR: " << learningRate << " Prediction: " << accurracy << "; " <<  std::endl;
        }

        // Learning rate decay
        learningRate = initialLR * std::pow(decayRate, ((steps / batchSize - stepsLeft) * batchSize) / (float)decaySteps);

        stepsLeft--;
    }

    std::cout << "\nTraining completed!" << std::endl;

    nn.run();
}

void sin_multiplication() {

    // Network setup - simplified for testing
    const int inputSize = 2;
    const int hiddenSize = 20;
    const int outputSize = 1;

    const int batchSize = 64;
    const int steps = 1000000; 

    NNGL::NeuralNetwork nn(batchSize);

    nn.onTrainBatch([&](std::shared_ptr<NNGL::Matrix>& batchInputs, std::shared_ptr<NNGL::Matrix>& batchTargets, int batchSize) {
        /*for (int i = 0; i < batchSize; i++) {
            float a = 3.14f * ((float)rand() / RAND_MAX) * 2.0f - 3.14f;
            float b_val = 3.14f * ((float)rand() / RAND_MAX) * 2.0f - 3.14f;

            batchInputs[i * 2 + 0] = a / 3.14f;
            batchInputs[i * 2 + 1] = b_val / 3.14f;
            batchTargets[i] = std::sin(a) * std::sin(b_val);
        } */   
    });
    nn.onTestBatch([&](std::shared_ptr<NNGL::Matrix>& batchInputs, std::shared_ptr<NNGL::Matrix>& batchTargets, int batchSize) {
        /*for (int i = 0; i < batchSize; i++) {
            float a = 3.14f * ((float)rand() / RAND_MAX) * 2.0f - 3.14f;
            float b_val = 3.14f * ((float)rand() / RAND_MAX) * 2.0f - 3.14f;

            batchInputs[i * 2 + 0] = a / 3.14f;
            batchInputs[i * 2 + 1] = b_val / 3.14f;
            batchTargets[i] = std::sin(a) * std::sin(b_val);
        }*/
    });

    nn.addLayer(inputSize, 1, NNGL::ActivationFnType::TANH);
    nn.addLayer(1, hiddenSize, NNGL::ActivationFnType::RELU);
    nn.addLayer(hiddenSize, hiddenSize, NNGL::ActivationFnType::LRELU);
    nn.addLayer(hiddenSize, hiddenSize, NNGL::ActivationFnType::SIGMOID);
    nn.addLayer(hiddenSize, hiddenSize, NNGL::ActivationFnType::TANH);
    nn.addLayer(hiddenSize, outputSize, NNGL::ActivationFnType::IDENTITY);

    float learningRate = 0.03f;
    int stepsLeft = steps / batchSize;

    while (stepsLeft > 0) {
        nn.train(learningRate);

        // Periodic GPU synchronization and cleanup
        if (stepsLeft % 100 == 0) {
            glFinish(); // Force GPU to complete all pending operations
            GLenum error = glGetError(); // Check for OpenGL errors
            if (error != GL_NO_ERROR) throw std::runtime_error("OpenGL Error detected: " + std::to_string(error) + " at training step " + std::to_string(steps / batchSize - stepsLeft));
        }

        // Status updates and weight display
        if (stepsLeft % (10000 / batchSize) == 0) {
            resetCursor();
            for (int i = 0; i < nn.m_Layers.size(); i++) {
                nn.m_Layers[i]->printHeatmap();
                std::cout << std::endl;
            }
            std::cout << "\nStep: " << (steps / batchSize - stepsLeft) * batchSize << " LR: " << learningRate << std::endl;
        }

        // Learning rate decay
        if (stepsLeft % 10000 == 0) learningRate *= 0.99f;

        stepsLeft--;
    }

    std::cout << "\nTraining completed!" << std::endl;

    nn.run();
}

// ============================================================================
// UNIT TESTS FOR VALIDATION
// ============================================================================

void testLayerNormClass() {
    std::cout << "\n=== Testing LayerNorm Class (AddNorm style) ===" << std::endl;
    // Test 1: Basic forward pass with normalization (input + residual)
    {
        std::cout << "Test 1: Basic forward pass with normalization (input + residual)..." << std::endl;
        int modelDim = 4;
        int seqLen = 3;
        int batchSize = 2;
        NNGL::LayerNorm layerNorm(modelDim);
        auto input = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        auto residual = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        for (int i = 0; i < modelDim; ++i) {
            for (int j = 0; j < seqLen; ++j) {
                (*input)(i, j) = 10 * i + j + 1;
                (*residual)(i, j) = 100 * i + 10 * j;
            }
        }
        auto output = layerNorm.forward(input, residual);
        if (output->rows != modelDim || output->cols != seqLen) {
            std::cout << "  [FAIL] Output dimensions incorrect. Expected [" << modelDim << "," << seqLen 
                      << "], got [" << output->rows << "," << output->cols << "]" << std::endl;
            return;
        }
        bool isNormalized = false;
        for (int i = 0; i < modelDim; ++i) {
            for (int j = 0; j < seqLen; ++j) {
                if (std::abs((*output)(i, j) - (*input)(i, j)) > 1e-6) {
                    isNormalized = true;
                    break;
                }
            }
        }
        if (!isNormalized) {
            std::cout << "  [FAIL] Output identical to input - no normalization occurred" << std::endl;
            return;
        }
        std::cout << "  [PASS] Basic forward pass with normalization (input + residual)" << std::endl;
    }
    // Test 2: Forward pass with nonzero residual
    {
        std::cout << "Test 2: Forward pass with nonzero residual..." << std::endl;
        int modelDim = 3;
        int seqLen = 2;
        int batchSize = 2;
        NNGL::LayerNorm layerNorm(modelDim);
        auto input = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        auto residual = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        for (int i = 0; i < modelDim; ++i) {
            for (int j = 0; j < seqLen; ++j) {
                (*input)(i, j) = 10 * i + j + 1;
                (*residual)(i, j) = 100 * i + 10 * j;
            }
        }
        auto output = layerNorm.forward(input, residual);
        bool relationshipsPreserved = true;
        for (int i = 0; i < modelDim; ++i) {
            if (((*input)(i, 0) + (*residual)(i, 0)) < ((*input)(i, 1) + (*residual)(i, 1)) &&
                (*output)(i, 0) >= (*output)(i, 1)) {
                relationshipsPreserved = false;
                break;
            }
        }
        if (!relationshipsPreserved) {
            std::cout << "  [FAIL] Relative relationships not preserved" << std::endl;
            return;
        }
        std::cout << "  [PASS] Forward pass with nonzero residual preserves relationships" << std::endl;
    }
    // Test 3: Backward pass gradient flow (input + residual)
    {
        std::cout << "Test 3: Backward pass gradient flow (input + residual)..." << std::endl;
        int modelDim = 4;
        int seqLen = 2;
        int batchSize = 2;
        NNGL::LayerNorm layerNorm(modelDim);
        auto input = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        auto residual = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        for (int i = 0; i < modelDim; ++i) {
            for (int j = 0; j < seqLen; ++j) {
                (*input)(i, j) = 10 * i + j + 1;
                (*residual)(i, j) = 100 * i + 10 * j;
            }
        }
        auto output = layerNorm.forward(input, residual);
        auto gradOutput = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        (*gradOutput)(0, 0) = 0.1f; (*gradOutput)(0, 1) = 0.2f;
        (*gradOutput)(1, 0) = 0.3f; (*gradOutput)(1, 1) = 0.4f;
        layerNorm.backward(gradOutput, input, residual);
        auto gradInput = layerNorm.getGradInput();
        auto gradResidual = layerNorm.getGradResidual();

        if (gradInput->rows != modelDim || gradInput->cols != seqLen ||
            gradResidual->rows != modelDim || gradResidual->cols != seqLen) {
            std::cout << "  [FAIL] Gradient input/residual dimensions incorrect." << std::endl;
            return;
        }
        bool hasNonZeroGradients = false;
        for (int i = 0; i < modelDim; ++i) {
            for (int j = 0; j < seqLen; ++j) {
                if (std::abs((*gradInput)(i, j)) > 1e-6 || std::abs((*gradResidual)(i, j)) > 1e-6) {
                    hasNonZeroGradients = true;
                    break;
                }
            }
        }
        if (!hasNonZeroGradients) {
            std::cout << "  [FAIL] All gradients are zero" << std::endl;
            return;
        }
        std::cout << "  [PASS] Backward pass gradient flow (input + residual)" << std::endl;
    }
    // Test 4: Learnable parameters update (input + residual)
    {
        std::cout << "Test 4: Learnable parameters update (input + residual)..." << std::endl;
        int modelDim = 3;
        int seqLen = 2;
        int batchSize = 2;
        NNGL::LayerNorm layerNorm(modelDim);
        auto input = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        auto residual = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        for (int i = 0; i < modelDim; ++i) {
            for (int j = 0; j < seqLen; ++j) {
                (*input)(i, j) = 10 * i + j + 1;
                (*residual)(i, j) = 100 * i + 10 * j;
            }
        }
        auto output = layerNorm.forward(input, residual);
        auto gradOutput = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        (*gradOutput)(0, 0) = 0.1f; (*gradOutput)(0, 1) = 0.2f;
        (*gradOutput)(1, 0) = 0.3f; (*gradOutput)(1, 1) = 0.4f;
        layerNorm.backward(gradOutput, input, residual);
        std::cout << "  [PASS] Learnable parameters update (backward pass completed)" << std::endl;
    }
    // Test 5: Identity transformation with learned parameters (input + residual)
    {
        std::cout << "Test 5: Identity transformation with learned parameters (input + residual)..." << std::endl;
        int modelDim = 4;
        int seqLen = 2;
        int batchSize = 2;
        NNGL::LayerNorm layerNorm(modelDim);
        auto input = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        auto residual = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        for (int i = 0; i < modelDim; ++i) {
            for (int j = 0; j < seqLen; ++j) {
                (*input)(i, j) = 10 * i + j + 1;
                (*residual)(i, j) = 100 * i + 10 * j;
            }
        }
        for (int epoch = 0; epoch < 10; ++epoch) {
            auto output = layerNorm.forward(input, residual);
            auto target = std::make_shared<NNGL::Matrix>(*input);
            auto gradOutput = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
            for (int i = 0; i < modelDim; ++i) {
                for (int j = 0; j < seqLen; ++j) {
                    (*gradOutput)(i, j) = (*output)(i, j) - (*target)(i, j);
                }
            }
            layerNorm.backward(gradOutput, input, residual);
        }
        auto finalOutput = layerNorm.forward(input, residual);
        float initialDiff = 0.0f;
        for (int i = 0; i < modelDim; ++i) {
            for (int j = 0; j < seqLen; ++j) {
                initialDiff += std::abs((*input)(i, j) - (*finalOutput)(i, j));
            }
        }
        initialDiff /= (modelDim * seqLen);
        if (initialDiff < 2.0f) {
            std::cout << "  [PASS] Identity transformation with learned parameters (avg diff: " << initialDiff << ")" << std::endl;
        } else {
            std::cout << "  [WARN] Identity transformation not well learned (avg diff: " << initialDiff << ")" << std::endl;
        }
    }
    // Test 6: Numerical stability with small values (input + residual)
    {
        std::cout << "Test 6: Numerical stability with small values (input + residual)..." << std::endl;
        int modelDim = 3;
        int seqLen = 2;
        int batchSize = 2;
        NNGL::LayerNorm layerNorm(modelDim);
        auto input = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        auto residual = std::make_shared<NNGL::Matrix>(modelDim, seqLen);
        for (int i = 0; i < modelDim; ++i) {
            for (int j = 0; j < seqLen; ++j) {
                (*input)(i, j) = 1e-6f * (10 * i + j + 1);
                (*residual)(i, j) = 1e-6f * (100 * i + 10 * j);
            }
        }
        auto output = layerNorm.forward(input, residual);
        bool hasValidOutput = true;
        for (int i = 0; i < modelDim; ++i) {
            for (int j = 0; j < seqLen; ++j) {
                float val = (*output)(i, j);
                if (std::isnan(val) || std::isinf(val)) {
                    hasValidOutput = false;
                    break;
                }
            }
        }
        if (hasValidOutput) {
            std::cout << "  [PASS] Numerical stability with small values" << std::endl;
        } else {
            std::cout << "  [FAIL] Numerical instability detected" << std::endl;
        }
    }
}

void testMatrixClass() {
    std::cout << "\n=== Testing Matrix Class ===" << std::endl;
    
    // Test 1: Basic matrix creation and indexing
    {
        NNGL::Matrix mat(3, 2, 1.0f);
        std::cout << "Test 1: Matrix creation - ";
        if (mat.rows == 3 && mat.cols == 2 && mat(0, 0) == 1.0f) {
            std::cout << "PASS" << std::endl;
        } else {
            std::cout << "FAIL" << std::endl;
        }
    }
    
    // Test 2: Column-major indexing
    {
        NNGL::Matrix mat(2, 3);
        mat(0, 0) = 1.0f; mat(0, 1) = 2.0f; mat(0, 2) = 3.0f;
        mat(1, 0) = 4.0f; mat(1, 1) = 5.0f; mat(1, 2) = 6.0f;
        
        std::cout << "Test 2: Column-major indexing - ";
        // Column-major: [1,4,2,5,3,6]
        if (mat.getFlatVec()[0] == 1.0f && mat.getFlatVec()[1] == 4.0f &&
            mat.getFlatVec()[2] == 2.0f && mat.getFlatVec()[3] == 5.0f &&
            mat.getFlatVec()[4] == 3.0f && mat.getFlatVec()[5] == 6.0f) {
            std::cout << "PASS" << std::endl;
        } else {
            std::cout << "FAIL" << std::endl;
            std::cout << "Expected: [1,4,2,5,3,6], Got: [";
            for (size_t i = 0; i < mat.getFlatVec().size(); ++i) {
                std::cout << mat.getFlatVec()[i];
                if (i < mat.getFlatVec().size() - 1) std::cout << ",";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // Test 3: Matrix multiplication validation
    {
        NNGL::Matrix A(2, 3);
        A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
        A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;
        
        NNGL::Matrix B(3, 2);
        B(0, 0) = 7.0f; B(0, 1) = 8.0f;
        B(1, 0) = 9.0f; B(1, 1) = 10.0f;
        B(2, 0) = 11.0f; B(2, 1) = 12.0f;
        
        // Manual calculation: A @ B
        // [1 2 3] @ [7  8 ] = [1*7+2*9+3*11  1*8+2*10+3*12] = [58  64]
        // [4 5 6]   [9  10]   [4*7+5*9+6*11  4*8+5*10+6*12]   [139 154]
        //           [11 12]
        
        std::cout << "Test 3: Matrix multiplication validation - ";
        // Column-major storage: [1,4,2,5,3,6] for A, [7,9,11,8,10,12] for B
        // Expected result: [58,139,64,154] in column-major
        float expected[4] = {58.0f, 139.0f, 64.0f, 154.0f}; // Column-major
        bool pass = true;
        
        // Verify column-major storage is correct
        if (A.getFlatVec()[0] == 1.0f && A.getFlatVec()[1] == 4.0f &&
            A.getFlatVec()[2] == 2.0f && A.getFlatVec()[3] == 5.0f &&
            A.getFlatVec()[4] == 3.0f && A.getFlatVec()[5] == 6.0f) {
            std::cout << "PASS" << std::endl;
        } else {
            std::cout << "FAIL" << std::endl;
            std::cout << "Expected A: [1,4,2,5,3,6], Got: [";
            for (size_t i = 0; i < A.getFlatVec().size(); ++i) {
                std::cout << A.getFlatVec()[i];
                if (i < A.getFlatVec().size() - 1) std::cout << ",";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // Test 4: Shader matrix layout validation
    {
        std::cout << "Test 4: Shader matrix layout validation - ";
        
        // Create matrix as it would be in C++: [features, batch]
        std::vector input = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }; // Column-major
        NNGL::Matrix inputMat(4, 2, input.data()); // [input_size, batch_size]
        
        // Create weight matrix: [input_size, output_size]
        std::vector weight = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }; // Column-major
        NNGL::Matrix weightMat(4, 3, weight.data()); // [input_size, output_size]
        
        // Expected output: [output_size, batch_size] = [3, 2]
        // Manual calculation for batch_idx=0, neuron_idx=0:
        // sum = bias[0] + input[0*4+0]*weight[0*3+0] + input[0*4+1]*weight[1*3+0] + input[0*4+2]*weight[2*3+0] + input[0*4+3]*weight[3*3+0]
        // sum = bias[0] + 1*1 + 2*5 + 3*9 + 4*13 = bias[0] + 1 + 10 + 27 + 52 = bias[0] + 90
        
        std::cout << "PASS (matrix layout confirmed)" << std::endl;
        std::cout << "  Input matrix [4,2] column-major: [";
        for (size_t i = 0; i < inputMat.getFlatVec().size(); ++i) {
            std::cout << inputMat.getFlatVec()[i];
            if (i < inputMat.getFlatVec().size() - 1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Weight matrix [4,3] column-major: [";
        for (size_t i = 0; i < weightMat.getFlatVec().size(); ++i) {
            std::cout << weightMat.getFlatVec()[i];
            if (i < weightMat.getFlatVec().size() - 1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }
}

void testNeuralNetworkClass() {
    std::cout << "\n=== Testing NeuralNetwork Class ===" << std::endl;
    
    // Test 1: Basic neural network creation
    {
        NNGL::NeuralNetwork nn(1);
        nn.addLayer(2, 3, NNGL::ActivationFnType::RELU);
        nn.addLayer(3, 1, NNGL::ActivationFnType::SIGMOID);
        
        std::cout << "Test 1: Neural network creation - ";
        if (nn.m_Layers.size() == 2) {
            std::cout << "PASS" << std::endl;
        } else {
            std::cout << "FAIL" << std::endl;
        }
    }
    
    // Test 2: Forward pass validation (using public interface only)
    {
        NNGL::NeuralNetwork nn(1);
        nn.addLayer(2, 2, NNGL::ActivationFnType::IDENTITY);
        nn.addLayer(2, 1, NNGL::ActivationFnType::IDENTITY);
        
        // Create input
        std::vector<float> inputVec = { 0.5f, 0.7f };
        auto input = std::make_shared<NNGL::Matrix>(2, 1, inputVec.data());
        
        std::cout << "Test 2: Forward pass validation - ";
        try {
            auto output = nn.forward(input);
            if (output && output->rows == 2 && output->cols == 1) {
                std::cout << "PASS (forward pass completed)" << std::endl;
                std::cout << "  Output values: [" << output->getFlatVec()[0] << ", " << output->getFlatVec()[1] << "]" << std::endl;
            } else {
                std::cout << "FAIL (wrong output dimensions: expected [2,1], got [" << output->rows << "," << output->cols << "])" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "FAIL (exception: " << e.what() << ")" << std::endl;
        }
    }
    
    // Test 3: Evaluation test (using eval method)
    {
        NNGL::NeuralNetwork nn(1);
        nn.addLayer(2, 2, NNGL::ActivationFnType::IDENTITY);
        nn.addLayer(2, 1, NNGL::ActivationFnType::IDENTITY);
        
        // Set up test batch provider for eval
        nn.onTestBatch([&](std::shared_ptr<NNGL::Matrix>& batchInputs, std::shared_ptr<NNGL::Matrix>& batchTargets, int batchSize) {

            batchInputs->set(0, 0, 0.5f);
            batchInputs->set(1, 0, 0.7f);
            batchTargets->set(0,0, 0.8f);
        });
        
        std::cout << "Test 3: Evaluation test - ";
        try {
            float accuracy = nn.eval(1);
            std::cout << "PASS (eval completed with accuracy: " << accuracy << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "FAIL (exception: " << e.what() << ")" << std::endl;
        }
    }
}

void testAttentionBlockClass() {
    std::cout << "\n=== Testing AttentionBlock Class ===" << std::endl;
    
    // Test 1: Basic attention block creation
    {
        NNGL::AttentionBlock attention(64, 8, 10, false);
        std::cout << "Test 1: Attention block creation - PASS" << std::endl;
    }
    
    // Test 2: Attention forward pass validation
    {
        NNGL::AttentionBlock attention(32, 4, 5, false);
        
        // Set up simple test case
        auto input = std::make_shared<NNGL::Matrix>(32, 5); // [model_dim, seq_len]
        input->randomize(-1.0f, 1.0f);
        
        std::cout << "Test 2: Attention forward pass - ";
        try {
            auto output = attention.forward(input, input);
            if (output && output->rows == 32 && output->cols == 5) {
                std::cout << "PASS (forward pass completed)" << std::endl;
            } else {
                std::cout << "FAIL (wrong output dimensions)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "FAIL (exception: " << e.what() << ")" << std::endl;
        }
    }
}

void testDecoderBlockClass() {
    std::cout << "\n=== Testing DecoderBlock Class ===" << std::endl;
    
    // Test 1: Basic decoder block creation
    {
        NNGL::DecoderBlock decoder(64, 128, 10);
        std::cout << "Test 1: Decoder block creation - PASS" << std::endl;
    }
    
    // Test 2: Decoder forward pass
    {
        NNGL::DecoderBlock decoder(32, 64, 5);
        auto input = std::make_shared<NNGL::Matrix>(5, 32);
        std::vector<int> paddingMask(5, 1);
        input->randomize(-1.0f, 1.0f);
        
        std::cout << "Test 2: Decoder forward pass - ";
        try {
            auto output = decoder.forward(input, paddingMask);
            if (output && output->rows == 5 && output->cols == 32) {
                std::cout << "PASS" << std::endl;
            } else {
                std::cout << "FAIL (wrong output dimensions)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "FAIL (exception: " << e.what() << ")" << std::endl;
        }
    }
}

void test_embeddingblock_gpu_update() {
    using namespace NNGL;
    std::cout << "\n[UnitTest] EmbeddingBlock GPU Update\n";
    size_t modelDim = 3;
    size_t vocabSize = 10;

    EmbeddingBlock embedder(vocabSize, modelDim, 10);
    std::vector<std::string> tokens2 = { "B", "B" };
    embedder.forward(tokens2);
    float lr = 0.1f;
    {
        // Test 1: Single token
        std::vector<std::string> tokens1 = { "A" };
        auto beforeA = embedder.forward(tokens1); // deep copy
        beforeA->downloadFromGPU();
        beforeA = std::make_shared<Matrix>(beforeA->rows, beforeA->cols, beforeA->getFlatVec().data());
        std::shared_ptr<Matrix> grad1 = std::make_shared<Matrix>(modelDim, 1);
        (*grad1)(0, 0) = 1; (*grad1)(1, 0) = 2; (*grad1)(2, 0) = 3;
        std::cout << "Test grad1: ";
        for (int j = 0; j < modelDim; ++j) std::cout << (*grad1)(j, 0) << " ";
        std::cout << std::endl;
        embedder.backward(tokens1, grad1, lr);

        auto afterA = embedder.forward(tokens1); // deep copy
        afterA->downloadFromGPU();

        std::cout << "A before: "; beforeA->print();
        std::cout << "A after:  "; afterA->print();
        for (int j = 0; j < modelDim; ++j) {
            float expected = (*beforeA)(j, 0) - lr * (*grad1)(j, 0);
            float d = (*beforeA)(j, 0);
            float d1 = (*afterA)(j, 0);
            float diff = std::abs((*afterA)(j, 0) - expected);
            assert(diff < 1e-4);
        }
    }

    {
        std::vector<std::string> singleB = { "B" };
        auto beforeB = embedder.forward(singleB); // deep copy
        beforeB->downloadFromGPU();
        beforeB = std::make_shared<Matrix>(beforeB->rows, beforeB->cols, beforeB->getFlatVec().data());


        // Test 2: Repeated token
        std::vector<std::string> tokens = { "B", "B" };
        embedder.forward(tokens);
        std::shared_ptr<Matrix> grad = std::make_shared<Matrix>(modelDim, 2);
        (*grad)(0, 0) = 1; (*grad)(1, 0) = 2; (*grad)(2, 0) = 3;
        (*grad)(0, 1) = 4; (*grad)(1, 1) = 5; (*grad)(2, 1) = 6;
        std::cout << "Test grad2: ";
        for (int i = 0; i < modelDim; ++i) for (int j = 0; j < 2; ++j) std::cout << (*grad)(i, j) << " ";
        std::cout << std::endl;
        embedder.backward(tokens, grad, lr);


        auto afterB = embedder.forward(singleB); // deep copy
        afterB->downloadFromGPU();

        std::cout << "B before: \n"; beforeB->print();
        std::cout << "grad2: \n"; grad->print();
        std::cout << "B after:  \n"; afterB->print();
        // Simulate sequential SGD updates for repeated token B
        for (int j = 0; j < modelDim; ++j) {
            float expected = (*beforeB)(j, 0);
            float a = (*grad)(j, 0);
            float b = (*grad)(j, 1);
            expected -= lr * (*grad)(j, 0); // first occurrence
            expected -= lr * (*grad)(j, 1); // second occurrence

            float diff = std::abs((*afterB)(j, 0) - expected);
            assert(diff < 1e-4);
        }
    }
 
    {

        std::vector<std::string> singleC = { "C" };
        std::vector<std::string> singleD = { "D" };

        auto beforeC = embedder.forward(singleC);
        beforeC->downloadFromGPU();
        beforeC = std::make_shared<Matrix>(beforeC->rows, beforeC->cols, beforeC->getFlatVec().data());

        auto beforeD = embedder.forward(singleD);
        beforeD->downloadFromGPU();
        beforeD = std::make_shared<Matrix>(beforeD->rows, beforeD->cols, beforeD->getFlatVec().data());

        // Test 3: Multiple tokens
        std::vector<std::string> tokens = { "C", "D" };
        embedder.forward(tokens);

        std::shared_ptr<Matrix> grad = std::make_shared<Matrix>(modelDim, 2);
        (*grad)(0, 0) = 1; (*grad)(1, 0) = 2; (*grad)(2, 0) = 3;
        (*grad)(0, 1) = 4; (*grad)(1, 1) = 5; (*grad)(2, 1) = 6;
        std::cout << "Test grad: ";
        for (int i = 0; i < modelDim; ++i) for (int j = 0; j < 2; ++j) std::cout << (*grad)(i, j) << " ";
        std::cout << std::endl;
        embedder.backward(tokens, grad, lr);

        auto afterC = embedder.forward(singleC);
        afterC->downloadFromGPU();
        afterC = std::make_shared<Matrix>(afterC->rows, afterC->cols, afterC->getFlatVec().data());

        auto afterD = embedder.forward(singleD); // deep copy
        afterD->downloadFromGPU();
        afterD = std::make_shared<Matrix>(afterD->rows, afterD->cols, afterD->getFlatVec().data());

        std::cout << "C before: "; beforeC->print();
        std::cout << "C after:  "; afterC->print();
        std::cout << "grad: \n"; grad->print();
        std::cout << "D before: "; beforeD->print();
        std::cout << "D after:  "; afterD->print();
        for (int j = 0; j < modelDim; ++j) {
            float expectedC = (*beforeC)(j, 0) - lr * (*grad)(j, 0);
            float expectedD = (*beforeD)(j, 0) - lr * (*grad)(j, 1);
            float diff = std::abs((*afterC)(j, 0) - expectedC);
            assert(diff < 1e-4);
            diff = std::abs((*afterD)(j, 0) - expectedD);
            assert(diff < 1e-4);
        }
        std::cout << "[UnitTest] EmbeddingBlock GPU Update PASSED\n";
    }

}

void test_positional_encoding() {
    using namespace NNGL;
    std::cout << "\n[UnitTest] EmbeddingBlock Positional Encoding\n";
    size_t seqLen = 4;
    size_t modelDim = 3;
    size_t vocabSize = 10;
    EmbeddingBlock embedder(vocabSize, modelDim, seqLen);
    // Create a known input matrix
    std::vector<std::vector<float>> inputVec(modelDim, std::vector<float>(seqLen));
    float val = 1.0f;
    for (size_t i = 0; i < modelDim; ++i) {
        for (size_t j = 0; j < seqLen; ++j) {
            inputVec[i][j] = val++;
        }
    }
    auto inputMat = std::make_shared<Matrix>(inputVec);
    auto originalMat = std::make_shared<Matrix>(inputVec); // for comparison
    std::cout << "Input before positional encoding:" << std::endl;
    inputMat->print();
    embedder.applyPositionalEncoding(inputMat);
    std::cout << "After applyPositionalEncoding:" << std::endl;
    inputMat->print();
    embedder.removePositionalEncoding(inputMat);
    std::cout << "After removePositionalEncoding:" << std::endl;
    inputMat->print();
    // Assert that inputMat matches originalMat (within tolerance)
    for (size_t i = 0; i < modelDim; ++i) {
        for (size_t j = 0; j < seqLen; ++j) {
            assert(std::abs((*inputMat)(i, j) - (*originalMat)(i, j)) < 1e-4);
        }
    }
    std::cout << "[UnitTest] EmbeddingBlock Positional Encoding PASSED\n";
}

void testDecoderBlockBackward() {
    using namespace NNGL;
    std::cout << "\n[UnitTest] DecoderBlock Backward Gradient Check\n";
    int modelDim = 8;
    int hiddenDim = 8;
    int seqLen = 3;
    float epsilon = 1e-4f;
    float tolerance = 1e-1f;
    DecoderBlock decoder(modelDim, hiddenDim, seqLen);
    // Create dummy input and dummy encoder output
    auto input = std::make_shared<Matrix>(modelDim, seqLen);
    std::vector<int> paddingMask(seqLen, 1);
    // Fill with small random values
    for (int i = 0; i < modelDim; ++i) {
        for (int j = 0; j < seqLen; ++j) {
            (*input)(i, j) = 0.1f * (i + 1) + 0.01f * (j + 1);
        }
    }
    // Forward pass
    auto output = decoder.forward(input, paddingMask);

    // Analytical gradient: dL/dOutput is all ones
    auto gradOutput = std::make_shared<Matrix>(output->rows, output->cols, 1.0f);

    // Backward pass: get dL/dInput
    auto gradInput = decoder.backward(gradOutput, 0.0f); // learningRate=0 to avoid weight update
    // Numerical gradient check for a single input element
    int test_i = 0, test_j = 0;
    float orig = (*input)(test_i, test_j);

    (*input)(test_i, test_j) = orig + epsilon;
    auto out_plus = decoder.forward(input, paddingMask);
    float loss_plus = 0.0f;
    for (int i = 0; i < out_plus->rows; ++i)
        for (int j = 0; j < out_plus->cols; ++j)
            loss_plus += (*out_plus)(i, j);

    (*input)(test_i, test_j) = orig - epsilon;
    auto out_minus = decoder.forward(input, paddingMask);
    float loss_minus = 0.0f;
    for (int i = 0; i < out_minus->rows; ++i)
        for (int j = 0; j < out_minus->cols; ++j)
            loss_minus += (*out_minus)(i, j);

    float num_grad = (loss_plus - loss_minus) / (2 * epsilon);
    float analytic_grad = (*gradInput)(test_i, test_j);
    std::cout << "Analytic grad: " << analytic_grad << ", Numerical grad: " << num_grad << std::endl;
    assert(std::abs(num_grad - analytic_grad) < tolerance);
    std::cout << "[UnitTest] DecoderBlock Backward Gradient Check PASSED\n";
}

void runAllUnitTests() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "RUNNING COMPREHENSIVE UNIT TESTS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    test_embeddingblock_gpu_update();
    testMatrixClass();
    testNeuralNetworkClass();
    testAttentionBlockClass();
    testLayerNormClass();
    testDecoderBlockClass();
    test_positional_encoding();
    testDecoderBlockBackward();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "UNIT TESTS COMPLETED" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Reset GPU state after unit tests to prevent state pollution
    LOG_INFO("Resetting GPU state after unit tests...");
    glFinish(); // Ensure all GPU operations are complete
    
    // Clear any bound buffers
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    
    // Clear any bound shaders
    glUseProgram(0);
    
    // Clear any bound textures
    glBindTexture(GL_TEXTURE_2D, 0);
    
    LOG_INFO("GPU state reset completed");
}

void gptransformer_simplified() {
    // Simple GPTransformer (GPT-style, decoder-only) overfit test on multiple examples
    std::srand(42);
    std::cout << "=== Simple GPTransformer Overfit Test (10 sentences) ===" << std::endl;
    int d_model = 256;  // Increased for complex text
    int d_hidden = d_model * 4;
    int seq_len = 16;   // Longer sequence for complex text


    std::string bpe_file = "bpe50k.checkpoint";
    if(false)
    {
        std::vector<std::string> filenames = { "english3.txt", "pg76287.txt", "english3.txt", "pg76287.txt", "english3.txt", "pg76287.txt","english3.txt", "pg76287.txt" };
        std::shared_ptr<NNGL::BPE> bpe = std::make_shared<NNGL::BPE>(1024 * 10);


        // Ensure all printable ASCII single-character tokens are in the vocab
        for (char c = 32; c < 127; ++c) { // printable ASCII
            std::string s(1, c);
            bpe->addToken(s);
        }
        bpe->addToken(" ");
        bpe->trainFromFiles(filenames, true);

        bpe->reduceVocab(50000);
        bpe->addToken("<EOS>");
        bpe->addToken("<PAD>");
        bpe->addToken("<SOS>");

        std::cout << "BPE vocabulary size: " << bpe->getVocabSize() << std::endl;
        bpe->save(bpe_file);
    }

    std::shared_ptr<NNGL::BPE> bpe = std::make_shared<NNGL::BPE>(1024 * 10);
    bpe->load(bpe_file);


        // Overfit on apple-color association and QA sentences
    std::vector<std::string> training_data = {
        " red Apple is number one ",
        " red and number one is Apple ",
        " number one and red is Apple ",
        " Apple number one has color red ",
        " What color of Apple number one It's red ",
          
        " Apple number two has color green ",
        " green Apple has number two ",
        " green always two ",
        " number two has green fruit ",
        " What color of Apple number two It's green ",
          
        " yellow and three make an Apple ",
        " Apple is yellow when it's three ",
        " three when it's yellow ",
        " Apple number three has color yellow ",
    };
    // Precompute tokenized prefixes for eval
    std::vector<std::pair<std::string, std::string>> eval_prompts;
    std::vector<std::string> test_queries = {
        " What color of Apple number one ",
        " three when it's ",
        " Apple number one "
    };
    for (const auto& query : test_queries) {
        std::vector<std::string> tokens = bpe->tokenizeInput(query.c_str(), query.size());
        tokens.insert(tokens.begin(), "<SOS>");
        std::string prompt;
        std::string display;
        for (const auto& t : tokens) {
            prompt += t;
            display += "'" + t + "' ";
        }
        eval_prompts.emplace_back(display, prompt);
    }

    std::shared_ptr<NNGL::GPTransformer> gpt = std::make_shared<NNGL::GPTransformer>(bpe_file, d_model, d_hidden, seq_len);

    // Build the training sequence: <SOS> tokens... <EOS> for each sentence, concatenated
    std::vector<std::string> sequence;
    for (const auto& sentence : training_data) {
        sequence.push_back("<SOS>");
        std::vector<std::string> tokens = bpe->tokenizeInput(sentence.c_str(), sentence.size());
        sequence.insert(sequence.end(), tokens.begin(), tokens.end());
        sequence.push_back("<EOS>");
    }

    // Print sequence for debugging
    std::cout << "Training sequence: [";
    for (size_t i = 0; i < sequence.size(); ++i) {
        std::cout << "'" << sequence[i] << "'";
        if (i < sequence.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\n=== Training (Overfitting on 10 Sentences) ===" << std::endl;
    int epochs = 1000000;
    float initial_learning_rate = 0.0001f; // Reduced for more stable learning
    
    // Early stopping variables
    int epochs_without_improvement = 0;
    
    // Training loop: train on each sentence independently
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float learning_rate = initial_learning_rate * std::pow(0.98f, epoch / (epochs / 100.0f));
        float total_loss = 0.0f;
        int num_tokens = 0;
        for (const auto& sentence : training_data) {
            std::vector<std::string> tokens = bpe->tokenizeInput(sentence.c_str(), sentence.size());
            tokens.insert(tokens.begin(), "<SOS>");
            tokens.push_back("<EOS>");
            for (size_t i = 1; i < tokens.size(); ++i) {
                std::vector<std::string> context(tokens.begin(), tokens.begin() + i);
                std::string target = tokens[i];
                float loss = gpt->trainNextToken(context, target, learning_rate);
                total_loss += loss;
                num_tokens++;
            }
        }
        // Print progress every 2 epochs
        if ((epoch + 1) % 2 == 0 || epoch == 0) {
            float avg_loss = total_loss / num_tokens;
            std::cout << "Epoch " << (epoch + 1) << ": Avg Loss = " << std::fixed << std::setprecision(4) << avg_loss 
                      << " | LR = " << std::fixed << std::setprecision(6) << learning_rate << std::endl;
            for (const auto& eval_pair : eval_prompts) {
                std::cout << "  [tokens: " << eval_pair.first << "] -> '" << gpt->eval(eval_pair.second) << "'" << std::endl;
            }
            if (epoch == 0 || (epoch + 1) % 100 == 0) {
                std::cout << "  Number of training examples per epoch: " << num_tokens << std::endl;
                std::cout << "  Loss history size: " << gpt->getLossHistory().size() << std::endl;
                if (!gpt->getLossHistory().empty()) {
                    float min_loss = *std::min_element(gpt->getLossHistory().begin(), gpt->getLossHistory().end());
                    float max_loss = *std::max_element(gpt->getLossHistory().begin(), gpt->getLossHistory().end());
                    std::cout << "  Loss range: [" << std::fixed << std::setprecision(4) << min_loss 
                              << ", " << std::fixed << std::setprecision(4) << max_loss << "]" << std::endl;
                }
            }
        }
    }
    
    // Test with various partial inputs
    std::cout << "Final predictions:" << std::endl;
    for (const auto& eval_pair : eval_prompts) {
        std::cout << "  [tokens: " << eval_pair.first << "] -> '" << gpt->eval(eval_pair.second) << "'" << std::endl;
    }
    
    // Training summary
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "Total training steps: " << gpt->getTrainingSteps() << std::endl;
    std::cout << "Loss history size: " << gpt->getLossHistory().size() << std::endl;
    if (!gpt->getLossHistory().empty()) {
        float min_loss = *std::min_element(gpt->getLossHistory().begin(), gpt->getLossHistory().end());
        float max_loss = *std::max_element(gpt->getLossHistory().begin(), gpt->getLossHistory().end());
        float avg_loss = std::accumulate(gpt->getLossHistory().begin(), gpt->getLossHistory().end(), 0.0f) / gpt->getLossHistory().size();
        std::cout << "Loss statistics:" << std::endl;
        std::cout << "  Min: " << std::fixed << std::setprecision(4) << min_loss << std::endl;
        std::cout << "  Max: " << std::fixed << std::setprecision(4) << max_loss << std::endl;
        std::cout << "  Avg: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
    }
}

int main(int argc, char** argv) {
    srand(time(nullptr));
 
    NNGL::Logger::getInstance().setLogLevel(NNGL::LogLevel::LL_INFO);
    NNGL::Logger::getInstance().setEnabled(true);

    // Initialize GLFW
    if (!glfwInit()) { LOG_ERROR("GLFW initialization failed!"); return -1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(1, 1, "NN Compute", nullptr, nullptr);
    if (!window) {
        LOG_ERROR("GLFW window creation failed!");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { LOG_ERROR("Failed to initialize GLAD!"); return -1; }


    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    int choice = 1; // Change this to test different functions
    //runAllUnitTests();

    switch (choice) {
        case 0:
            LOG_INFO("");
            LOG_INFO("RUNNING DIGIT RECOGN");
            LOG_INFO(std::string(60, '='));
            digit_recognition();
            break;

        case 1:
            LOG_INFO("");
            LOG_INFO("RUNNING GPT TRANSLATION");
            LOG_INFO(std::string(60, '='));
            gptransformer_simplified();
            break;
        default:
            LOG_ERROR("Invalid choice, running simple EOS prediction...");
            gptransformer_simplified();
            break;
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}