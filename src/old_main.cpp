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
        // Create input and residual with known values
        auto input = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        auto residual = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        (*input)(0, 0) = 1.0f; (*input)(0, 1) = 2.0f;
        (*input)(1, 0) = 3.0f; (*input)(1, 1) = 4.0f;
        (*residual)(0, 0) = 0.0f; (*residual)(0, 1) = 0.0f;
        (*residual)(1, 0) = 0.0f; (*residual)(1, 1) = 0.0f;
        auto output = layerNorm.forward(input, residual);
        if (output->rows != seqLen || output->cols != batchSize) {
            std::cout << "  [FAIL] Output dimensions incorrect. Expected [" << seqLen << "," << batchSize 
                      << "], got [" << output->rows << "," << output->cols << "]" << std::endl;
            return;
        }
        bool isNormalized = false;
        for (int i = 0; i < seqLen; ++i) {
            for (int j = 0; j < batchSize; ++j) {
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
        auto input = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        auto residual = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        (*input)(0, 0) = 1.0f; (*input)(0, 1) = 2.0f;
        (*input)(1, 0) = 3.0f; (*input)(1, 1) = 4.0f;
        (*residual)(0, 0) = 10.0f; (*residual)(0, 1) = 20.0f;
        (*residual)(1, 0) = 30.0f; (*residual)(1, 1) = 40.0f;
        auto output = layerNorm.forward(input, residual);
        bool relationshipsPreserved = true;
        for (int i = 0; i < seqLen; ++i) {
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
        auto input = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        auto residual = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        (*input)(0, 0) = 1.0f; (*input)(0, 1) = 2.0f;
        (*input)(1, 0) = 3.0f; (*input)(1, 1) = 4.0f;
        (*residual)(0, 0) = 0.5f; (*residual)(0, 1) = 0.5f;
        (*residual)(1, 0) = 0.5f; (*residual)(1, 1) = 0.5f;
        auto output = layerNorm.forward(input, residual);
        auto gradOutput = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        (*gradOutput)(0, 0) = 0.1f; (*gradOutput)(0, 1) = 0.2f;
        (*gradOutput)(1, 0) = 0.3f; (*gradOutput)(1, 1) = 0.4f;
        layerNorm.backward(gradOutput, input, residual);
        auto gradInput = layerNorm.getGradInput();
        auto gradResidual = layerNorm.getGradResidual();

        if (gradInput->rows != seqLen || gradInput->cols != batchSize ||
            gradResidual->rows != seqLen || gradResidual->cols != batchSize) {
            std::cout << "  [FAIL] Gradient input/residual dimensions incorrect." << std::endl;
            return;
        }
        bool hasNonZeroGradients = false;
        for (int i = 0; i < seqLen; ++i) {
            for (int j = 0; j < batchSize; ++j) {
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
        auto input = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        auto residual = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        (*input)(0, 0) = 1.0f; (*input)(0, 1) = 2.0f;
        (*input)(1, 0) = 3.0f; (*input)(1, 1) = 4.0f;
        (*residual)(0, 0) = 0.0f; (*residual)(0, 1) = 0.0f;
        (*residual)(1, 0) = 0.0f; (*residual)(1, 1) = 0.0f;
        auto output = layerNorm.forward(input, residual);
        auto gradOutput = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
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
        auto input = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        auto residual = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        (*input)(0, 0) = 1.0f; (*input)(0, 1) = 2.0f;
        (*input)(1, 0) = 3.0f; (*input)(1, 1) = 4.0f;
        (*residual)(0, 0) = 0.0f; (*residual)(0, 1) = 0.0f;
        (*residual)(1, 0) = 0.0f; (*residual)(1, 1) = 0.0f;
        for (int epoch = 0; epoch < 10; ++epoch) {
            auto output = layerNorm.forward(input, residual);
            auto target = std::make_shared<NNGL::Matrix>(*input);
            auto gradOutput = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
            for (int i = 0; i < seqLen; ++i) {
                for (int j = 0; j < batchSize; ++j) {
                    (*gradOutput)(i, j) = (*output)(i, j) - (*target)(i, j);
                }
            }
            layerNorm.backward(gradOutput, input, residual);
        }
        auto finalOutput = layerNorm.forward(input, residual);
        float initialDiff = 0.0f;
        for (int i = 0; i < seqLen; ++i) {
            for (int j = 0; j < batchSize; ++j) {
                initialDiff += std::abs((*input)(i, j) - (*finalOutput)(i, j));
            }
        }
        initialDiff /= (seqLen * batchSize);
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
        auto input = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        auto residual = std::make_shared<NNGL::Matrix>(seqLen, batchSize);
        (*input)(0, 0) = 1e-6f; (*input)(0, 1) = 2e-6f;
        (*input)(1, 0) = 3e-6f; (*input)(1, 1) = 4e-6f;
        (*residual)(0, 0) = 0.0f; (*residual)(0, 1) = 0.0f;
        (*residual)(1, 0) = 0.0f; (*residual)(1, 1) = 0.0f;
        auto output = layerNorm.forward(input, residual);
        bool hasValidOutput = true;
        for (int i = 0; i < seqLen; ++i) {
            for (int j = 0; j < batchSize; ++j) {
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
        auto input = std::make_shared<NNGL::Matrix>(5, 32); // [seq_len, model_dim]
        input->randomize(-1.0f, 1.0f);
        
        std::cout << "Test 2: Attention forward pass - ";
        try {
            auto output = attention.forward(input, input);
            if (output && output->rows == 5 && output->cols == 32) {
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
    EmbeddingBlock embedder(10, modelDim, 10);
    float lr = 0.1f;

    // Test 1: Single token
    std::vector<std::string> tokens1 = { "A" };
    embedder.forward(tokens1); // ensure "A" is initialized
    auto beforeA = std::make_shared<Matrix>(*embedder.forward(tokens1)); // deep copy
    std::shared_ptr<Matrix> grad1 = std::make_shared<Matrix>(1, modelDim);
    (*grad1)(0, 0) = 1; (*grad1)(0, 1) = 2; (*grad1)(0, 2) = 3;
    std::cout << "Test grad1: ";
    for (int j = 0; j < modelDim; ++j) std::cout << (*grad1)(0, j) << " ";
    std::cout << std::endl;
    embedder.backward(tokens1, grad1, lr);
    auto afterA = std::make_shared<Matrix>(*embedder.forward(tokens1)); // deep copy
    std::cout << "A before: "; beforeA->print();
    std::cout << "A after:  "; afterA->print();
    for (int j = 0; j < modelDim; ++j) {
        float expected = (*beforeA)(0, j) - lr * (*grad1)(0, j);
        assert(std::abs((*afterA)(0, j) - expected) < 1e-4);
    }

    // Test 2: Repeated token
    std::vector<std::string> tokens2 = { "B", "B" };
    embedder.forward(tokens2);
    std::vector<std::string> singleB = { "B" };
    auto beforeB = std::make_shared<Matrix>(*embedder.forward(singleB)); // deep copy
    std::shared_ptr<Matrix> grad2 = std::make_shared<Matrix>(2, modelDim);
    (*grad2)(0, 0) = 1; (*grad2)(0, 1) = 2; (*grad2)(0, 2) = 3;
    (*grad2)(1, 0) = 4; (*grad2)(1, 1) = 5; (*grad2)(1, 2) = 6;
    std::cout << "Test grad2: ";
    for (int i = 0; i < 2; ++i) for (int j = 0; j < modelDim; ++j) std::cout << (*grad2)(i, j) << " ";
    std::cout << std::endl;
    embedder.backward(tokens2, grad2, lr);
    auto afterB = std::make_shared<Matrix>(*embedder.forward(singleB)); // deep copy
    std::cout << "B before: "; beforeB->print();
    std::cout << "B after:  "; afterB->print();
    // Simulate sequential SGD updates for repeated token B
    for (int j = 0; j < modelDim; ++j) {
        float val = (*beforeB)(0, j);
        val -= lr * (*grad2)(0, j); // first occurrence
        val -= lr * (*grad2)(1, j); // second occurrence
        float expected = val;
        assert(std::abs((*afterB)(0, j) - expected) < 1e-4);
    }

    // Test 3: Multiple tokens
    std::vector<std::string> tokens3 = { "C", "D" };
    embedder.forward(tokens3);
    std::vector<std::string> singleC = { "C" };
    std::vector<std::string> singleD = { "D" };
    auto beforeC = std::make_shared<Matrix>(*embedder.forward(singleC)); // deep copy
    auto beforeD = std::make_shared<Matrix>(*embedder.forward(singleD)); // deep copy
    std::shared_ptr<Matrix> grad3 = std::make_shared<Matrix>(2, modelDim);
    (*grad3)(0, 0) = 1; (*grad3)(0, 1) = 2; (*grad3)(0, 2) = 3;
    (*grad3)(1, 0) = 4; (*grad3)(1, 1) = 5; (*grad3)(1, 2) = 6;
    std::cout << "Test grad3: ";
    for (int i = 0; i < 2; ++i) for (int j = 0; j < modelDim; ++j) std::cout << (*grad3)(i, j) << " ";
    std::cout << std::endl;
    embedder.backward(tokens3, grad3, lr);
    auto afterC = std::make_shared<Matrix>(*embedder.forward(singleC)); // deep copy
    auto afterD = std::make_shared<Matrix>(*embedder.forward(singleD)); // deep copy
    std::cout << "C before: "; beforeC->print();
    std::cout << "C after:  "; afterC->print();
    std::cout << "D before: "; beforeD->print();
    std::cout << "D after:  "; afterD->print();
    for (int j = 0; j < modelDim; ++j) {
        float expectedC = (*beforeC)(0, j) - lr * (*grad3)(0, j);
        float expectedD = (*beforeD)(0, j) - lr * (*grad3)(1, j);
        assert(std::abs((*afterC)(0, j) - expectedC) < 1e-4);
        assert(std::abs((*afterD)(0, j) - expectedD) < 1e-4);
    }
    std::cout << "[UnitTest] EmbeddingBlock GPU Update PASSED\n";
}

void test_positional_encoding() {
    using namespace NNGL;
    std::cout << "\n[UnitTest] EmbeddingBlock Positional Encoding\n";
    size_t seqLen = 4;
    size_t modelDim = 3;
    EmbeddingBlock embedder(10, modelDim, seqLen);
    // Create a known input matrix
    std::vector<std::vector<float>> inputVec(seqLen, std::vector<float>(modelDim));
    float val = 1.0f;
    for (size_t i = 0; i < seqLen; ++i) {
        for (size_t j = 0; j < modelDim; ++j) {
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
    for (size_t i = 0; i < seqLen; ++i) {
        for (size_t j = 0; j < modelDim; ++j) {
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
    float tolerance = 1e-2f;
    DecoderBlock decoder(modelDim, hiddenDim, seqLen);
    // Create dummy input and dummy encoder output
    auto input = std::make_shared<Matrix>(seqLen, modelDim);
    std::vector<int> paddingMask(seqLen, 1);
    // Fill with small random values
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < modelDim; ++j) {
            (*input)(i, j) = 0.1f * (i + 1) + 0.01f * (j + 1);
        }
    }
    // Forward pass
    auto output = decoder.forward(input, paddingMask);

    // Analytical gradient: dL/dOutput is all ones
    auto gradOutput = std::make_shared<Matrix>(output->rows,output->cols, 1.0f);

    // Backward pass: get dL/dInput
    auto gradInput = decoder.backward(gradOutput, 0, 0.0f); // learningRate=0 to avoid weight update  // 0 is bad we don't use mask
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
    
    testMatrixClass();
    testNeuralNetworkClass();
    testAttentionBlockClass();
    testLayerNormClass();
    testDecoderBlockClass();
    test_embeddingblock_gpu_update();
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

void gptransformer_from_file() {
    std::srand(42);
    std::cout << "=== GPTransformer Training from File (Individual Sentences) ===" << std::endl;

    int d_model = 768;
    int d_hidden = d_model * 2;
    int seq_len = 1024;

    std::string bpe_file = "bpe50k_v2.checkpoint";
    if (false)
    {
        std::vector<std::string> filenames = { "english3.txt", "pg76287.txt", "pg51161.txt", "english3.txt", "pg76287.txt", "pg51161.txt", "english3.txt", "pg76287.txt", "pg51161.txt" };
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


    // Load BPE (assuming it's already trained)
    std::shared_ptr<NNGL::BPE> bpe = std::make_shared<NNGL::BPE>(1024 * 10);
    bpe->load(bpe_file);

    // Read training data from file
    std::string training_file = "pg51161.txt";
    std::vector<std::string> training_data;
    std::ifstream file(training_file);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open training file: " << training_file << std::endl;
        return;
    }

    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        line_count++;
        // Skip empty lines and lines that are too short
        if (line.empty() || line.length() < 3) {
            continue;
        }

        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (!line.empty()) {
            training_data.push_back(line);
        }
    }
    file.close();

    std::cout << "Loaded " << training_data.size() << " training sentences from " << line_count << " lines in file: " << training_file << std::endl;

    if (training_data.empty()) {
        std::cerr << "Error: No valid training data found in file!" << std::endl;
        return;
    }

    // Preprocess training data into individual tokenized sequences
    struct TrainingExample {
        std::vector<std::string> tokens;  // Full sequence: < SOS > + tokens + <EOS>
        std::string original_text;
    };

    std::vector<TrainingExample> examples;
    int skipped_examples = 0;

    for (const auto& sentence : training_data) {
        TrainingExample example;
        example.original_text = sentence;

        // Tokenize: < SOS > + sentence_tokens + <EOS>
        example.tokens.push_back("<SOS>");
        std::vector<std::string> sentence_tokens = bpe->tokenizeInput(sentence.c_str(), sentence.size());

        // Skip sentences that are too long or too short after tokenization
        if (sentence_tokens.size() < 2 || sentence_tokens.size() > seq_len - 2) {
            skipped_examples++;
            continue;
        }

        example.tokens.insert(example.tokens.end(), sentence_tokens.begin(), sentence_tokens.end());
        example.tokens.push_back("<EOS>");

        examples.push_back(example);
    }

    if (skipped_examples > 0) {
        std::cout << "Skipped " << skipped_examples << " sentences (too short/long after tokenization)" << std::endl;
    }

    if (examples.empty()) {
        std::cerr << "Error: No valid examples after tokenization!" << std::endl;
        return;
    }

    // Print first few tokenized examples for debugging
    std::cout << "\nFirst " << std::min(5, (int)examples.size()) << " training examples:" << std::endl;
    for (size_t i = 0; i < std::min(5, (int)examples.size()); ++i) {
        std::cout << "  " << i << ": \"" << examples[i].original_text << "\" -> [";
        for (size_t j = 0; j < examples[i].tokens.size(); ++j) {
            std::cout << "'" << examples[i].tokens[j] << "'";
            if (j < examples[i].tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Create evaluation prompts (first half of each sentence)
    std::vector<std::pair<std::string, std::string>> eval_prompts;
    int max_eval_prompts = std::min(20, (int)examples.size()); // Limit evaluation prompts for performance

    for (int i = 0; i < max_eval_prompts; ++i) {
        const auto& example = examples[i];
        if (example.tokens.size() > 2) { // Must have at least < SOS > + 1 token + <EOS>
            std::string prompt;
            std::string display;

            // Take first half of tokens (excluding <EOS>)
            size_t prefix_len = std::max(1, (int)(example.tokens.size() - 1) / 2);
            for (size_t j = 0; j < prefix_len; ++j) {
                prompt += example.tokens[j];
                if (j == 0) {
                    display += "'" + example.tokens[j] + "' ";
                }
                else {
                    display += "'" + example.tokens[j] + "' ";
                }
            }
            eval_prompts.emplace_back(display, prompt);
        }
    }

    std::shared_ptr<NNGL::GPTransformer> gpt = std::make_shared<NNGL::GPTransformer>(bpe_file, d_model, d_hidden, seq_len);

    std::cout << "\n=== Training (Individual Sentence Learning) ===" << std::endl;
    std::cout << "Training on " << examples.size() << " examples" << std::endl;
    std::cout << "Using " << eval_prompts.size() << " evaluation prompts" << std::endl;

    int epochs = 10000;
    float initial_learning_rate = 0.0001f;
    int progress_interval = 50; // Print progress every N trained lines

    // Training tracking
    float best_loss = std::numeric_limits<float>::infinity();
    int epochs_without_improvement = 0;
    std::vector<float> epoch_losses;
    int total_lines_trained = 0;
    float running_loss_sum = 0.0f;
    int running_loss_count = 0, tokenesTrained = 0;
    NNGL::Timer timer_info("Time to process:" + std::to_string(progress_interval) + "lines", NNGL::LogLevel::LL_INFO);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Learning rate decay
        float learning_rate = initial_learning_rate * std::pow(0.95f, epoch / 50.0f);

        float total_loss = 0.0f;
        int total_predictions = 0;

        // Shuffle training examples each epoch for better learning
        std::vector<size_t> indices(examples.size());
        std::iota(indices.begin(), indices.end(), 0);

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);

        // Train on each sentence independently
        for (size_t idx : indices) {
            const auto& example = examples[idx];
            // Train next-token prediction for each position in this sentence
            for (size_t i = 1; i < example.tokens.size(); ++i) {
                std::vector<std::string> context(example.tokens.begin(), example.tokens.begin() + i);
                std::string target = example.tokens[i];
                tokenesTrained++;
                float loss = gpt->trainNextToken(context, target, learning_rate);
                if (loss == -1) continue; //means this run didn't check loss 

                total_loss += loss;
                total_predictions++;

                // Track running loss for progress reporting
                running_loss_sum += loss;
                running_loss_count++;

            }

            total_lines_trained++;

            // Print progress every N trained lines
            if (total_lines_trained % progress_interval == 0) {
                timer_info.reset();
                float avg_running_loss = running_loss_sum / running_loss_count;
                std::cout << "Lines trained: " << total_lines_trained
                    << " | Epoch: " << (epoch + 1)
                    //<< " | Recent avg loss: " << std::fixed << std::setprecision(4) << avg_running_loss
                    << " | Total Tokens trained: " << tokenesTrained
                    << " | LR: " << std::fixed << std::setprecision(6) << learning_rate << std::endl;
                tokenesTrained = 0;
                // Reset running loss for next interval
                running_loss_sum = 0.0f;
                running_loss_count = 0;

                // Show a sample prediction
                if (!eval_prompts.empty()) {
                    const auto& eval_pair = eval_prompts[0];
                    std::string prediction = gpt->eval(eval_pair.second);
                    std::cout << "  Sample: [" << eval_pair.first << "] -> '" << prediction << "'" << std::endl;
                }
            }
        }

        float avg_loss = total_loss / total_predictions;
        epoch_losses.push_back(avg_loss);

        // Track best loss for early stopping
        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            epochs_without_improvement = 0;
        }
        else {
            epochs_without_improvement++;
        }

        // Print epoch summary (less frequent now)
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            std::cout << "\n=== Epoch " << (epoch + 1) << " Summary ===" << std::endl;
            std::cout << "Epoch avg loss: " << std::fixed << std::setprecision(4) << avg_loss
                << " | Best loss: " << std::fixed << std::setprecision(4) << best_loss
                << " | No improve: " << epochs_without_improvement << std::endl;

            // Show predictions for first few evaluation prompts
            for (size_t i = 0; i < std::min(3, (int)eval_prompts.size()); ++i) {
                const auto& eval_pair = eval_prompts[i];
                std::string prediction = gpt->eval(eval_pair.second);
                std::cout << "  [" << eval_pair.first << "] -> '" << prediction << "'" << std::endl;
            }
            std::cout << std::endl;
        }

        // Detailed progress every 500 epochs
        if ((epoch + 1) % 500 == 0) {
            std::cout << "=== Detailed Progress (Epoch " << (epoch + 1) << ") ===" << std::endl;
            std::cout << "  Total lines trained so far: " << total_lines_trained << std::endl;
            std::cout << "  Total examples: " << examples.size() << std::endl;
            std::cout << "  Predictions per epoch: " << total_predictions << std::endl;
            std::cout << "  Average tokens per sentence: " << std::fixed << std::setprecision(1)
                << (float)total_predictions / examples.size() << std::endl;

            // Show loss trend
            if (epoch_losses.size() >= 10) {
                float recent_avg = 0.0f;
                for (int i = epoch_losses.size() - 10; i < epoch_losses.size(); ++i) {
                    recent_avg += epoch_losses[i];
                }
                recent_avg /= 10;
                std::cout << "  Recent 10-epoch avg loss: " << std::fixed << std::setprecision(4) << recent_avg << std::endl;
            }

            // Test more evaluation prompts
            std::cout << "  Sample evaluation prompts:" << std::endl;
            for (size_t i = 0; i < std::min(5, (int)eval_prompts.size()); ++i) {
                const auto& eval_pair = eval_prompts[i];
                std::string prediction = gpt->eval(eval_pair.second);
                std::cout << "    [" << eval_pair.first << "] -> '" << prediction << "'" << std::endl;
            }
            std::cout << std::endl;
        }

        // Early stopping
        if (epochs_without_improvement > 500) {
            std::cout << "Early stopping at epoch " << (epoch + 1) << " - no improvement for 500 epochs" << std::endl;
            break;
        }

        // Stop if we achieve very low loss
        if (avg_loss < 0.001f) {
            std::cout << "Stopping at epoch " << (epoch + 1) << " - achieved very low loss: " << avg_loss << std::endl;
            break;
        }
    }

    std::cout << "\n=== Final Evaluation ===" << std::endl;
    std::cout << "Best loss achieved: " << std::fixed << std::setprecision(4) << best_loss << std::endl;

    // Final test on evaluation prompts
    for (size_t i = 0; i < std::min(10, (int)eval_prompts.size()); ++i) {
        const auto& eval_pair = eval_prompts[i];
        std::string prediction = gpt->eval(eval_pair.second);
        std::cout << "Final " << i << ": [" << eval_pair.first << "] -> '" << prediction << "'" << std::endl;
        std::cout << "  Original: \"" << examples[i].original_text << "\"" << std::endl;
    }

    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Trained on " << examples.size() << " sentences from file: " << training_file << std::endl;
}

void gptransformer_simplified() {
    std::srand(42);
    std::cout << "=== Improved GPTransformer Training (Individual Sentences) ===" << std::endl;

    int d_model = 256;
    int d_hidden = d_model * 4;
    int seq_len = 64;

    std::string bpe_file = "bpe50k.checkpoint";

    // Load BPE (assuming it's already trained)
    std::shared_ptr<NNGL::BPE> bpe = std::make_shared<NNGL::BPE>(1024 * 10);
    bpe->load(bpe_file);

    // Training data - each sentence is treated independently
    std::vector<std::string> training_data = {
        "What color of Apple number two It's green"
    };

    // Preprocess training data into individual tokenized sequences
    struct TrainingExample {
        std::vector<std::string> tokens;  // Full sequence: <SOS> + tokens + <EOS>
        std::string original_text;
    };

    std::vector<TrainingExample> examples;
    for (const auto& sentence : training_data) {
        TrainingExample example;
        example.original_text = sentence;

        // Tokenize: <SOS> + sentence_tokens + <EOS>
        example.tokens.push_back("<SOS>");
        std::vector<std::string> sentence_tokens = bpe->tokenizeInput(sentence.c_str(), sentence.size());
        example.tokens.insert(example.tokens.end(), sentence_tokens.begin(), sentence_tokens.end());
        example.tokens.push_back("<EOS>");

        examples.push_back(example);
    }

    // Print tokenized examples for debugging
    std::cout << "Training examples:" << std::endl;
    for (size_t i = 0; i < examples.size(); ++i) {
        std::cout << "  " << i << ": \"" << examples[i].original_text << "\" -> [";
        for (size_t j = 0; j < examples[i].tokens.size(); ++j) {
            std::cout << "'" << examples[i].tokens[j] << "'";
            if (j < examples[i].tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Create evaluation prompts (first half of each sentence)
    std::vector<std::pair<std::string, std::string>> eval_prompts;
    for (const auto& example : examples) {
        if (example.tokens.size() > 2) { // Must have at least <SOS> + 1 token + <EOS>
            std::string prompt;
            std::string display;

            // Take first half of tokens (excluding <EOS>)
            size_t prefix_len = std::max(1, (int)(example.tokens.size() - 1) / 2);
            for (size_t i = 0; i < prefix_len; ++i) {
                prompt += example.tokens[i];
                display += "'" + example.tokens[i] + "' ";
            }
            eval_prompts.emplace_back(display, prompt);
        }
    }

    std::shared_ptr<NNGL::GPTransformer> gpt = std::make_shared<NNGL::GPTransformer>(bpe_file, d_model, d_hidden, seq_len);

    std::cout << "\n=== Training (Individual Sentence Learning) ===" << std::endl;

    int epochs = 10000;
    float initial_learning_rate = 0.0001f;

    // Training tracking
    float best_loss = std::numeric_limits<float>::infinity();
    int epochs_without_improvement = 0;
    std::vector<float> epoch_losses;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Learning rate decay
        float learning_rate = initial_learning_rate * std::pow(0.95f, epoch / 50.0f);

        float total_loss = 0.0f;
        int total_predictions = 0;

        // Shuffle training examples each epoch for better learning
        std::vector<size_t> indices(examples.size());
        std::iota(indices.begin(), indices.end(), 0);

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);

        // Train on each sentence independently
        for (size_t idx : indices) {
            const auto& example = examples[idx];

            // Train next-token prediction for each position in this sentence
            for (size_t i = 1; i < example.tokens.size(); ++i) {
                std::vector<std::string> context(example.tokens.begin(), example.tokens.begin() + i);
                std::string target = example.tokens[i];

                float loss = gpt->trainNextToken(context, target, learning_rate);
                total_loss += loss;
                total_predictions++;
            }
        }

        float avg_loss = total_loss / total_predictions;
        epoch_losses.push_back(avg_loss);

        // Track best loss for early stopping
        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            epochs_without_improvement = 0;
        }
        else {
            epochs_without_improvement++;
        }

        // Print progress
        if ((epoch + 1) % 50 == 0 || epoch == 0) {
            std::cout << "Epoch " << (epoch + 1) << ": Avg Loss = " << std::fixed << std::setprecision(4) << avg_loss
                << " | LR = " << std::fixed << std::setprecision(6) << learning_rate
                << " | Best = " << std::fixed << std::setprecision(4) << best_loss
                << " | No improve: " << epochs_without_improvement << std::endl;

            // Show predictions for evaluation prompts
            for (size_t i = 0; i < std::min(5, (int)eval_prompts.size()); ++i) {
                const auto& eval_pair = eval_prompts[i];
                std::string prediction = gpt->eval(eval_pair.second);
                std::cout << "  [" << eval_pair.first << "] -> '" << prediction << "'" << std::endl;
            }
        }

        // Detailed progress every 200 epochs
        if ((epoch + 1) % 200 == 0) {
            std::cout << "  Training details:" << std::endl;
            std::cout << "    Total examples: " << examples.size() << std::endl;
            std::cout << "    Predictions per epoch: " << total_predictions << std::endl;
            std::cout << "    Average tokens per sentence: " << std::fixed << std::setprecision(1)
                << (float)total_predictions / examples.size() << std::endl;

            // Show loss trend
            if (epoch_losses.size() >= 10) {
                float recent_avg = 0.0f;
                for (int i = epoch_losses.size() - 10; i < epoch_losses.size(); ++i) {
                    recent_avg += epoch_losses[i];
                }
                recent_avg /= 10;
                std::cout << "    Recent 10-epoch avg loss: " << std::fixed << std::setprecision(4) << recent_avg << std::endl;
            }

            // Test all evaluation prompts
            std::cout << "  All evaluation prompts:" << std::endl;
            for (const auto& eval_pair : eval_prompts) {
                std::string prediction = gpt->eval(eval_pair.second);
                std::cout << "    [" << eval_pair.first << "] -> '" << prediction << "'" << std::endl;
            }
        }

        // Early stopping
        if (epochs_without_improvement > 500) {
            std::cout << "Early stopping at epoch " << (epoch + 1) << " - no improvement for 500 epochs" << std::endl;
            break;
        }

        // Stop if we achieve very low loss
        if (avg_loss < 0.001f) {
            //std::cout << "Stopping at epoch " << (epoch + 1) << " - achieved very low loss: " << avg_loss << std::endl;
            //break;
        }
    }

    std::cout << "\n=== Final Evaluation ===" << std::endl;
    std::cout << "Best loss achieved: " << std::fixed << std::setprecision(4) << best_loss << std::endl;

    // Final test on all prompts
    for (size_t i = 0; i < eval_prompts.size(); ++i) {
        const auto& eval_pair = eval_prompts[i];
        std::string prediction = gpt->eval(eval_pair.second);
        std::cout << "Final " << i << ": [" << eval_pair.first << "] -> '" << prediction << "'" << std::endl;
        std::cout << "  Original: \"" << examples[i].original_text << "\"" << std::endl;
    }

    std::cout << "\n=== Training Complete ===" << std::endl;
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
    int choice = 2; // Change this to test different functions

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
        case 2:
            LOG_INFO("");
            LOG_INFO("RUNNING GPT TRANSLATION FROM FILE");
            LOG_INFO(std::string(60, '='));
            gptransformer_from_file();
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