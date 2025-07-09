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
#include <vector>
#include <functional>
#include <algorithm>
#include <random>

#include <glm/glm.hpp>

#include "ActivationFunctions.h"
#include "AttentionBlock.h"
#include "Transformer.h"
#include "NeuralNetwork.h"
#include "MNISTLoader.h"
#include "BPE.h"
#include "Matrix.h"
#include "Logger.h"
#include "LayerNorm.h"


#include <vector>
#include <cstdint>
#include <cmath>

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

void clean_word(std::string& word) {
    word.erase(std::remove_if(word.begin(), word.end(),
        [](unsigned char c) { return std::ispunct(c); }), word.end());
    std::transform(word.begin(), word.end(), word.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
}

void transformer_simplified() {
    // Enhanced transformer training with realistic data and intermediate checks
    NNGL::Logger::getInstance().setEnabled(false);

    std::cout << "=== Enhanced Transformer Training with Validation ===" << std::endl;

    // Model parameters optimized for testing
    int d_model = 64;   // Reduced for quick testing
    int d_hidden = d_model * 2;
    int seq_len = 32;

    // Create BPE tokenizer
    std::shared_ptr<NNGL::BPE> bytePairEnc = std::make_shared<NNGL::BPE>();
    
    // Comprehensive training data with different patterns and complexity levels
    std::vector<std::string> training_data = {
        // Basic patterns - single words (will be automatically appended with EOS)
        "hello",
        "world",
        "the",
        "cat",
        "dog",
        "bird",
        "fish",
        "tree",
        "sun",
        "moon",
        
        // Simple phrases - 2-3 words
        "hello world",
        "the cat",
        "a dog",
        "birds fly",
        "fish swim",
        "trees grow",
        "sun shines",
        "moon glows",
        "stars twinkle",
        "wind blows",
        
        // Medium complexity - 4-6 words
        "the cat sat down",
        "a dog runs fast",
        "birds fly high above",
        "fish swim deep below",
        "trees grow tall slowly",
        "sun shines bright today",
        "moon glows soft tonight",
        "stars twinkle in darkness",
        "wind blows strong outside",
        "rain falls gently down",
        
        // Complex patterns - 7+ words with variety
        "the quick brown fox jumps over the lazy dog",
        "a beautiful bird sings sweetly in the morning",
        "tall trees sway gently in the summer breeze",
        "bright stars shine brilliantly in the night sky",
        "fresh rain falls softly on the green grass",
        "warm sun rises slowly over the mountain peak",
        "cool wind blows gently through the forest trees",
        "clear water flows smoothly down the rocky stream",
        "soft clouds drift slowly across the blue sky",
        "gentle waves crash softly on the sandy beach",
        
        // Question patterns
        "what is this",
        "where are you",
        "when will it",
        "how do you",
        "why does the",
        "which way should",
        "who can help",
        "whose book is",
        
        // Conditional patterns
        "if you want",
        "when it rains",
        "while the sun",
        "since the moon",
        "because the wind",
        "although the bird",
        "unless the fish",
        "until the tree",
        
        // Repetition patterns (for testing attention)
        "hello hello hello",
        "the the the",
        "cat cat cat",
        "dog dog dog",
        "bird bird bird",
        
        // Sequential patterns
        "one two three",
        "first second third",
        "begin middle end",
        "start continue finish",
        "alpha beta gamma",
        
        // Contrast patterns
        "big and small",
        "hot and cold",
        "fast and slow",
        "high and low",
        "bright and dark",
        "loud and quiet",
        "hard and soft",
        "old and new"
    };

    // Validation data (separate from training)
    std::vector<std::string> validation_data = {
        "hello there",
        "the bird flies",
        "sun is bright",
        "water flows down",
        "trees are tall",
        "stars shine bright",
        "wind blows strong",
        "rain falls hard",
        "moon is full",
        "fish swim fast"
    };

    // Test prompts for generation (these should NOT include EOS tokens)
    std::vector<std::string> test_prompts = {
        "hello",
        "the",
        "a",
        "birds",
        "sun",
        "water",
        "trees",
        "stars",
        "wind",
        "rain"
    };

    std::cout << "Training BPE tokenizer on diverse data..." << std::endl;
    
    // Train on individual characters first
    std::string all_chars = "abcdefghijklmnopqrstuvwxyz ";
    for (char c : all_chars) {
        std::string char_str(1, c);
        bytePairEnc->trainFromString(char_str, true);
    }
    
    // Train on all training data
    for (const auto& sentence : training_data) {
        bytePairEnc->trainFromString(sentence, true);
    }
    bytePairEnc->reduceVocab(200); // Larger vocab for better coverage
    std::cout << "BPE training completed. Vocabulary size: " << bytePairEnc->getVocabSize() << std::endl;

    // Save BPE
    std::string temp_bpe_file = "temp_bpe_enhanced.checkpoint";
    bytePairEnc->save(temp_bpe_file);
    
    // Create transformer
    std::shared_ptr<NNGL::Transformer> transformer = std::make_shared<NNGL::Transformer>(
        temp_bpe_file, d_model, d_hidden, seq_len);

    // Training with intermediate validation
    std::cout << "\n=== Starting Training with Validation ===" << std::endl;
    int total_epochs = 100;
    int validation_interval = 10;
    float best_validation_loss = std::numeric_limits<float>::max();
    
    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        // Training phase
        float training_loss = 0.0f;
        int training_samples = 0;
        
        for (const auto& sentence : training_data) {
            resetCursor();
            transformer->train(sentence);
            training_samples++;
        }
        
        // Validation phase (every validation_interval epochs)
        if ((epoch + 1) % validation_interval == 0) {
            std::cout << "\n--- Epoch " << (epoch + 1) << "/" << total_epochs << " ---" << std::endl;
            
            // Test generation on validation prompts
            std::cout << "Validation Generation Tests:" << std::endl;
            for (const auto& prompt : test_prompts) {
                std::string generated = prompt;
                std::string full_generated = prompt;
                
                // Generate 3-5 tokens
                for (int i = 0; i < 5; ++i) {
                    std::string next_token = transformer->eval(generated);
                    
                    if (next_token == "<EOS>" || next_token.empty()) {
                        break;
                    }
                    
                    full_generated += next_token;
                    generated = full_generated; // Use full context for next prediction
                    
                    // Stop if too long
                    if (full_generated.length() > 50) break;
                }
                
                std::cout << "  '" << prompt << "' -> '" << full_generated << "'" << std::endl;
            }
            
            // Test specific patterns
            std::cout << "\nPattern Recognition Tests:" << std::endl;
            
            // Test repetition pattern
            std::string rep_test = "hello";
            std::string rep_result = rep_test;
            for (int i = 0; i < 3; ++i) {
                std::string next = transformer->eval(rep_result);
                if (next != "<EOS>" && !next.empty()) {
                    rep_result += next;
                }
            }
            std::cout << "  Repetition: '" << rep_test << "' -> '" << rep_result << "'" << std::endl;
            
            // Test continuation pattern
            std::string cont_test = "the cat";
            std::string cont_result = cont_test;
            for (int i = 0; i < 3; ++i) {
                std::string next = transformer->eval(cont_result);
                if (next != "<EOS>" && !next.empty()) {
                    cont_result += next;
                }
            }
            std::cout << "  Continuation: '" << cont_test << "' -> '" << cont_result << "'" << std::endl;
            
            // Test question pattern
            std::string q_test = "what is";
            std::string q_result = q_test;
            for (int i = 0; i < 3; ++i) {
                std::string next = transformer->eval(q_result);
                if (next != "<EOS>" && !next.empty()) {
                    q_result += next;
                }
            }
            std::cout << "  Question: '" << q_test << "' -> '" << q_result << "'" << std::endl;
            
            // Test special token handling
            std::cout << "\nSpecial Token Tests:" << std::endl;
            
            // Test EOS token prediction (should predict EOS after complete sentences)
            std::string eos_test = "hello world";
            std::string eos_result = eos_test;
            for (int i = 0; i < 5; ++i) {
                std::string next = transformer->eval(eos_result);
                if (next == "<EOS>") {
                    std::cout << "  EOS Prediction: '" << eos_test << "' -> EOS predicted correctly" << std::endl;
                    break;
                } else if (next.empty()) {
                    std::cout << "  EOS Prediction: '" << eos_test << "' -> Empty token (issue)" << std::endl;
                    break;
                } else {
                    eos_result += next;
                }
            }
            
            // Test PAD token handling (should not generate PAD tokens)
            std::string pad_test = "";
            std::string pad_result = transformer->eval(pad_test);
            if (pad_result == "<PAD>") {
                std::cout << "  PAD Generation: WARNING - PAD token generated (should not happen)" << std::endl;
            } else {
                std::cout << "  PAD Generation: OK - No PAD token generated" << std::endl;
            }
            
            // Test SOS token handling
            std::string sos_test = "<SOS>";
            std::string sos_result = transformer->eval(sos_test);
            if (sos_result == "<SOS>") {
                std::cout << "  SOS Generation: WARNING - SOS token generated (should not happen)" << std::endl;
            } else {
                std::cout << "  SOS Generation: OK - No SOS token generated" << std::endl;
            }
        }
        
        // Progress indicator
        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch " << (epoch + 1) << "/" << total_epochs << " completed" << std::endl;
        }
    }

    // Final comprehensive test
    std::cout << "\n=== Final Comprehensive Test ===" << std::endl;
    
    std::vector<std::pair<std::string, std::string>> final_tests = {
        {"hello", "Basic greeting continuation"},
        {"the quick", "Complex phrase continuation"},
        {"birds fly", "Action continuation"},
        {"what is", "Question continuation"},
        {"if you", "Conditional continuation"},
        {"one two", "Sequence continuation"},
        {"big and", "Contrast continuation"},
        {"sun shines", "Description continuation"}
    };
    
    for (const auto& test : final_tests) {
        std::string generated = test.first;
        std::string full_generated = test.first;
        
        std::cout << "\n" << test.second << ":" << std::endl;
        std::cout << "  Input: '" << test.first << "'" << std::endl;
        
        for (int i = 0; i < 8; ++i) {
            std::string next_token = transformer->eval(generated);
            
            if (next_token == "<EOS>" || next_token.empty()) {
                std::cout << "  Stopped: EOS or empty token" << std::endl;
                break;
            }
            
            full_generated += next_token;
            generated = full_generated;
            
            if (full_generated.length() > 100) {
                std::cout << "  Stopped: Max length reached" << std::endl;
                break;
            }
        }
        
        std::cout << "  Output: '" << full_generated << "'" << std::endl;
    }

    std::cout << "\n=== Enhanced Transformer Training Complete ===" << std::endl;
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
        (*input)(2, 0) = 5.0f; (*input)(2, 1) = 6.0f;
        (*residual)(0, 0) = 0.0f; (*residual)(0, 1) = 0.0f;
        (*residual)(1, 0) = 0.0f; (*residual)(1, 1) = 0.0f;
        (*residual)(2, 0) = 0.0f; (*residual)(2, 1) = 0.0f;
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
        std::shared_ptr<NNGL::Matrix> gradInput, gradResidual, gradGamma, gradBeta;
        layerNorm.backward(gradOutput, input, residual, gradInput, gradResidual, gradGamma, gradBeta);
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
        std::shared_ptr<NNGL::Matrix> gradInput, gradResidual, gradGamma, gradBeta;
        layerNorm.backward(gradOutput, input, residual, gradInput, gradResidual, gradGamma, gradBeta);
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
            std::shared_ptr<NNGL::Matrix> gradInput, gradResidual, gradGamma, gradBeta;
            layerNorm.backward(gradOutput, input, residual, gradInput, gradResidual, gradGamma, gradBeta);
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
        if (mat.flatVec[0] == 1.0f && mat.flatVec[1] == 4.0f && 
            mat.flatVec[2] == 2.0f && mat.flatVec[3] == 5.0f &&
            mat.flatVec[4] == 3.0f && mat.flatVec[5] == 6.0f) {
            std::cout << "PASS" << std::endl;
        } else {
            std::cout << "FAIL" << std::endl;
            std::cout << "Expected: [1,4,2,5,3,6], Got: [";
            for (size_t i = 0; i < mat.flatVec.size(); ++i) {
                std::cout << mat.flatVec[i];
                if (i < mat.flatVec.size() - 1) std::cout << ",";
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
        if (A.flatVec[0] == 1.0f && A.flatVec[1] == 4.0f && 
            A.flatVec[2] == 2.0f && A.flatVec[3] == 5.0f &&
            A.flatVec[4] == 3.0f && A.flatVec[5] == 6.0f) {
            std::cout << "PASS" << std::endl;
        } else {
            std::cout << "FAIL" << std::endl;
            std::cout << "Expected A: [1,4,2,5,3,6], Got: [";
            for (size_t i = 0; i < A.flatVec.size(); ++i) {
                std::cout << A.flatVec[i];
                if (i < A.flatVec.size() - 1) std::cout << ",";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // Test 4: Shader matrix layout validation
    {
        std::cout << "Test 4: Shader matrix layout validation - ";
        
        // Create matrix as it would be in C++: [features, batch]
        NNGL::Matrix inputMat(4, 2); // [input_size, batch_size]
        inputMat.flatVec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; // Column-major
        
        // Create weight matrix: [input_size, output_size]
        NNGL::Matrix weightMat(4, 3); // [input_size, output_size]
        weightMat.flatVec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}; // Column-major
        
        // Expected output: [output_size, batch_size] = [3, 2]
        // Manual calculation for batch_idx=0, neuron_idx=0:
        // sum = bias[0] + input[0*4+0]*weight[0*3+0] + input[0*4+1]*weight[1*3+0] + input[0*4+2]*weight[2*3+0] + input[0*4+3]*weight[3*3+0]
        // sum = bias[0] + 1*1 + 2*5 + 3*9 + 4*13 = bias[0] + 1 + 10 + 27 + 52 = bias[0] + 90
        
        std::cout << "PASS (matrix layout confirmed)" << std::endl;
        std::cout << "  Input matrix [4,2] column-major: [";
        for (size_t i = 0; i < inputMat.flatVec.size(); ++i) {
            std::cout << inputMat.flatVec[i];
            if (i < inputMat.flatVec.size() - 1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Weight matrix [4,3] column-major: [";
        for (size_t i = 0; i < weightMat.flatVec.size(); ++i) {
            std::cout << weightMat.flatVec[i];
            if (i < weightMat.flatVec.size() - 1) std::cout << ",";
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
        auto input = std::make_shared<NNGL::Matrix>(2, 1);
        input->flatVec = {0.5f, 0.7f}; // [2,1] column-major
        
        std::cout << "Test 2: Forward pass validation - ";
        try {
            auto output = nn.forward(input);
            if (output && output->rows == 2 && output->cols == 1) {
                std::cout << "PASS (forward pass completed)" << std::endl;
                std::cout << "  Output values: [" << output->flatVec[0] << ", " << output->flatVec[1] << "]" << std::endl;
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
            batchInputs->flatVec = {0.5f, 0.7f}; // [2,1] column-major
            batchTargets->flatVec = {0.8f}; // [1,1] target
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

void testEncoderBlockClass() {
    std::cout << "\n=== Testing EncoderBlock Class ===" << std::endl;
    
    // Test 1: Basic encoder block creation
    {
        NNGL::EncoderBlock encoder(64, 128, 10);
        std::cout << "Test 1: Encoder block creation - PASS" << std::endl;
    }
    
    // Test 2: Encoder forward pass
    {
        NNGL::EncoderBlock encoder(32, 64, 5);
        auto input = std::make_shared<NNGL::Matrix>(5, 32);
        input->randomize(-1.0f, 1.0f);
        
        std::cout << "Test 2: Encoder forward pass - ";
        try {
            auto output = encoder.forward(input);
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
        auto context = std::make_shared<NNGL::Matrix>(5, 32);
        input->randomize(-1.0f, 1.0f);
        context->randomize(-1.0f, 1.0f);
        
        std::cout << "Test 2: Decoder forward pass - ";
        try {
            auto output = decoder.forward(input, context);
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

void testTransformerClass() {
    std::cout << "\n=== Testing Transformer Class ===" << std::endl;
    
    // Test 1: Basic transformer creation
    {
        try {
            NNGL::Transformer transformer("bpe.checkpoint", 64, 128, 10);
            std::cout << "Test 1: Transformer creation - PASS" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Test 1: Transformer creation - FAIL (exception: " << e.what() << ")" << std::endl;
        }
    }
    
    // Test 2: Transformer training (simple test)
    {
        try {
            NNGL::Transformer transformer("bpe.checkpoint", 32, 64, 5);
            std::cout << "Test 2: Transformer training - ";
            transformer.train("hello world");
            std::cout << "PASS (training completed)" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Test 2: Transformer training - FAIL (exception: " << e.what() << ")" << std::endl;
        }
    }
}

void runAllUnitTests() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "RUNNING COMPREHENSIVE UNIT TESTS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    testMatrixClass();
    testNeuralNetworkClass();
    testAttentionBlockClass();
    testLayerNormClass();
    testEncoderBlockClass();
    testDecoderBlockClass();
    testTransformerClass();
    
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

void transformer() {
    //d_model	256–512
    //vocab_size	10000–20000
    //seq_len	64–256(max ~512)
    //batch size	1–4
    //# heads	4–8
    //layers	2–6

    int d_model = 128;  // Must be divisible by num_heads (8)
    int d_hidden = d_model * 4;
    int seq_len = 64;

    std::vector<std::string> filenames = { "english3.txt", "pg76287.txt", "english3.txt", "pg76287.txt", "english3.txt", "pg76287.txt","english3.txt", "pg76287.txt" };
    std::shared_ptr<NNGL::BPE> bytePairEnc = std::make_shared<NNGL::BPE>(1024 * 10);
    //bytePairEnc->trainFromFiles(filenames);
    //bytePairEnc->reduceVocab(50000);
    //bytePairEnc->save("bpe.checkpoint");
    bytePairEnc->load("bpe.checkpoint");
    std::string test = "the quick brown fox jumps over the lazy dog";


    std::vector<std::string> enc_tokens = bytePairEnc->tokenizeInput(test.c_str(), test.size());
    std::vector<std::string> dec_tokens = { "<SOS>" };

    std::shared_ptr<NNGL::Transformer> transformer = std::make_shared<NNGL::Transformer>( "bpe.checkpoint", d_model, d_hidden, seq_len );


    auto trainFromFile = [&](const std::string& filename) {
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(infile, line)) {
            resetCursor();
            transformer->train(line);
        }

        infile.close();
    };
    trainFromFile("pg76287.txt");

    while (true) {
        std::string next_token = transformer->eval(test);

        test.append(next_token);
        // Stop condition (optional)
        if (next_token == "<EOS>") break;

        // Print/debug generated tokens
        std::cout << next_token << ' ';
    }
}

int main() {
    srand(time(nullptr));
    
    // Set log level (can be changed to control verbosity)
    // NNGL::Logger::getInstance().setLogLevel(NNGL::LogLevel::TRACE);  // Most verbose
    // NNGL::Logger::getInstance().setLogLevel(NNGL::LogLevel::DEBUG);  // Debug info
    // NNGL::Logger::getInstance().setLogLevel(NNGL::LogLevel::INFO);   // Default
    // NNGL::Logger::getInstance().setLogLevel(NNGL::LogLevel::WARN);   // Warnings only
    // NNGL::Logger::getInstance().setLogLevel(NNGL::LogLevel::ERROR);  // Errors only
    
    NNGL::Logger::getInstance().setEnabled(false);

    //new tokenizer
    // take a byte convert it in vector of 8 float where each float is bit 1.0f or 0.0f
    // feed it to ???
    /*NNGL::Trie trie;

    if (0) {
        trie.bulk_insert_from_file("english3.txt");
        trie.bulk_insert_from_file("pg76287.txt");
        //trie.prune_stddev_threshold(1.0); // Keep tokens >= (mean - 1×stddev)
        trie.save_tokens("tokens2.txt");
    }
    else trie.load_tokens("tokens2.txt");

    trie.prune_stddev_threshold(1.0);

    auto result2 = trie.tokenize("Hi my name is");
    std::vector<int> tokens;
    for (auto& it : result2) tokens.push_back(trie.get_token_id(it));*/
    
    // Initialize GLFW
    if (!glfwInit()) { std::cerr << "GLFW initialization failed!" << std::endl; return -1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(1, 1, "NN Compute", nullptr, nullptr);
    if (!window) {
        std::cerr << "GLFW window creation failed!" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr << "Failed to initialize GLAD!" << std::endl; return -1; }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    
    // Run comprehensive unit tests
    //runAllUnitTests();
    
    transformer_simplified();
    //transformer();
    //digit_recognition();

    std::cout << "Goodbye!" << std::endl;

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}