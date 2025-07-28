#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <cstdlib>
#include <ctime>
#include <algorithm>

// Include your headers (assuming these exist)
#include "BPE.h"
#include "Logger.h"

// GLFW and OpenGL headers
#define GLAD_GLX 0
extern "C" {
#include <glad/glad.h>
}

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

struct Config {
    std::string mode = "train";           // train, tokenize, info, reduce
    std::string checkpoint_path;          // Path to BPE checkpoint file
    std::vector<std::string> input_files; // Input text files for training
    std::string input_text;               // Direct text input for tokenization
    std::string output_file;              // Output file for tokenization results
    std::string output_checkpoint;        // Output checkpoint for reduce mode
    bool append_mode = false;             // Whether to append to existing checkpoint
    size_t vocab_size = 50000;            // Target vocabulary size
    size_t merge_limit = 10240;           // Merge limit for BPE training
    bool add_special_tokens = true;       // Add EOS, PAD, SOS tokens
    bool add_ascii_tokens = true;         // Add printable ASCII tokens
    bool verbose = false;                 // Verbose output
    bool help = false;                    // Show help
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Modes:\n";
    std::cout << "  --mode <train|tokenize|info|reduce>  Operation mode (default: train)\n\n";

    std::cout << "Common Options:\n";
    std::cout << "  --checkpoint <path>           Path to BPE checkpoint file\n";
    std::cout << "  --help                        Show this help message\n";
    std::cout << "  --verbose                     Enable verbose output\n\n";

    std::cout << "Training Mode Options:\n";
    std::cout << "  --input <file1,file2,...>     Input text files (comma-separated)\n";
    std::cout << "  --append                      Append to existing checkpoint\n";
    std::cout << "  --vocab-size <size>           Target vocabulary size (default: 50000)\n";
    std::cout << "  --merge-limit <limit>         Merge limit for BPE (default: 10240)\n";
    std::cout << "  --no-special-tokens           Don't add special tokens (EOS, PAD, SOS)\n";
    std::cout << "  --no-ascii-tokens             Don't add printable ASCII tokens\n\n";

    std::cout << "Tokenization Mode Options:\n";
    std::cout << "  --text <text>                 Text to tokenize (use quotes for spaces)\n";
    std::cout << "  --input <file>                Input text file to tokenize\n";
    std::cout << "  --output <file>               Output file for tokenization results\n\n";

    std::cout << "Reduce Mode Options:\n";
    std::cout << "  --vocab-size <size>           Target vocabulary size (required)\n";
    std::cout << "  --output <path>               Output checkpoint path (optional, defaults to input)\n\n";

    std::cout << "Examples:\n";
    std::cout << "  # Train new BPE model\n";
    std::cout << "  " << program_name << " --mode train --checkpoint model.bpe --input file1.txt,file2.txt\n\n";
    std::cout << "  # Append training to existing model\n";
    std::cout << "  " << program_name << " --mode train --checkpoint model.bpe --input new_data.txt --append\n\n";
    std::cout << "  # Tokenize text\n";
    std::cout << "  " << program_name << " --mode tokenize --checkpoint model.bpe --text \"Hello world!\"\n\n";
    std::cout << "  # Tokenize file\n";
    std::cout << "  " << program_name << " --mode tokenize --checkpoint model.bpe --input input.txt --output tokens.txt\n\n";
    std::cout << "  # Show model info\n";
    std::cout << "  " << program_name << " --mode info --checkpoint model.bpe\n\n";
    std::cout << "  # Reduce vocabulary size\n";
    std::cout << "  " << program_name << " --mode reduce --checkpoint model.bpe --vocab-size 10000\n\n";
    std::cout << "  # Reduce and save to new file\n";
    std::cout << "  " << program_name << " --mode reduce --checkpoint model.bpe --vocab-size 10000 --output small_model.bpe\n";
}

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : str) {
        if (c == delimiter) {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        }
        else {
            token += c;
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }
    return tokens;
}

bool parseArguments(int argc, char** argv, Config& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            config.help = true;
            return true;
        }
        else if (arg == "--mode" && i + 1 < argc) {
            config.mode = argv[++i];
            if (config.mode != "train" && config.mode != "tokenize" &&
                config.mode != "info" && config.mode != "reduce") {
                std::cerr << "Error: Invalid mode '" << config.mode << "'. Use train, tokenize, info, or reduce.\n";
                return false;
            }
        }
        else if (arg == "--checkpoint" && i + 1 < argc) {
            config.checkpoint_path = argv[++i];
        }
        else if (arg == "--input" && i + 1 < argc) {
            std::string input_str = argv[++i];
            if (config.mode == "train" || config.mode == "") {
                config.input_files = split(input_str, ',');
            }
            else {
                config.input_files = { input_str }; // Single file for tokenization
            }
        }
        else if (arg == "--text" && i + 1 < argc) {
            config.input_text = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            std::string output_str = argv[++i];
            if (config.mode == "reduce") {
                config.output_checkpoint = output_str;
            }
            else {
                config.output_file = output_str;
            }
        }
        else if (arg == "--append") {
            config.append_mode = true;
        }
        else if (arg == "--vocab-size" && i + 1 < argc) {
            config.vocab_size = std::stoul(argv[++i]);
        }
        else if (arg == "--merge-limit" && i + 1 < argc) {
            config.merge_limit = std::stoul(argv[++i]);
        }
        else if (arg == "--no-special-tokens") {
            config.add_special_tokens = false;
        }
        else if (arg == "--no-ascii-tokens") {
            config.add_ascii_tokens = false;
        }
        else if (arg == "--verbose" || arg == "-v") {
            config.verbose = true;
        }
        else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            return false;
        }
    }
    return true;
}

bool validateConfig(const Config& config) {
    if (config.checkpoint_path.empty()) {
        std::cerr << "Error: Checkpoint path is required (--checkpoint)\n";
        return false;
    }

    if (config.mode == "train") {
        if (config.input_files.empty()) {
            std::cerr << "Error: Input files required for training mode (--input)\n";
            return false;
        }

        // Check if input files exist
        for (const auto& file : config.input_files) {
            if (!std::filesystem::exists(file)) {
                std::cerr << "Error: Input file '" << file << "' does not exist\n";
                return false;
            }
        }

        // Check if checkpoint exists when not appending
        if (!config.append_mode && std::filesystem::exists(config.checkpoint_path)) {
            std::cout << "Warning: Checkpoint file '" << config.checkpoint_path
                << "' already exists and will be overwritten (use --append to append)\n";
        }
    }
    else if (config.mode == "tokenize") {
        if (config.input_text.empty() && config.input_files.empty()) {
            std::cerr << "Error: Either --text or --input required for tokenization mode\n";
            return false;
        }

        if (!config.input_files.empty() && !std::filesystem::exists(config.input_files[0])) {
            std::cerr << "Error: Input file '" << config.input_files[0] << "' does not exist\n";
            return false;
        }

        if (!std::filesystem::exists(config.checkpoint_path)) {
            std::cerr << "Error: Checkpoint file '" << config.checkpoint_path << "' does not exist\n";
            return false;
        }
    }
    else if (config.mode == "info") {
        if (!std::filesystem::exists(config.checkpoint_path)) {
            std::cerr << "Error: Checkpoint file '" << config.checkpoint_path << "' does not exist\n";
            return false;
        }
    }
    else if (config.mode == "reduce") {
        if (!std::filesystem::exists(config.checkpoint_path)) {
            std::cerr << "Error: Checkpoint file '" << config.checkpoint_path << "' does not exist\n";
            return false;
        }

        if (config.vocab_size == 0 || config.vocab_size == 50000) {
            std::cerr << "Error: Target vocabulary size must be specified for reduce mode (--vocab-size)\n";
            return false;
        }

        // If no output checkpoint specified, use input checkpoint (overwrite)
        if (config.output_checkpoint.empty()) {
            // This will be handled in reduceMode function
        }
    }

    return true;
}

bool initializeOpenGL() {
    if (!glfwInit()) {
        LOG_ERROR("GLFW initialization failed!");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(1, 1, "NN Compute", nullptr, nullptr);
    if (!window) {
        LOG_ERROR("GLFW window creation failed!");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        LOG_ERROR("Failed to initialize GLAD!");
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }

    return true;
}

void cleanupOpenGL() {
    glfwTerminate();
}

int trainMode(const Config& config) {
    MLGL::Timer timer("Training BPE model");
    LOG_INFO("Training BPE model...");

    auto bpe = std::make_shared<MLGL::BPE>(config.merge_limit);

    // Load existing checkpoint if appending
    if (config.append_mode && std::filesystem::exists(config.checkpoint_path)) {
        LOG_INFO("Loading existing checkpoint: " + config.checkpoint_path);
        try {
            bpe->load(config.checkpoint_path);
            LOG_INFO("Loaded checkpoint with vocabulary size: " + std::to_string(bpe->getVocabSize()));
        }
        catch (const std::exception& e) {
            LOG_ERROR("Error loading checkpoint: " + std::string(e.what()));
            return -1;
        }
    }

    // Add ASCII tokens if requested
    if (config.add_ascii_tokens) {
        LOG_INFO("Adding printable ASCII tokens...");
        for (char c = 32; c < 127; ++c) {
            std::string s(1, c);
            bpe->addToken(s);
        }
        bpe->addToken(" ");
    }

    // Train on input files
    LOG_INFO("Training on " + std::to_string(config.input_files.size()) + " files...");
    try {
        bpe->trainFromFiles(config.input_files, config.append_mode || config.add_ascii_tokens);
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error during training: " + std::string(e.what()));
        return -1;
    }

    // Reduce vocabulary if requested
    if (config.vocab_size > 0 && bpe->getVocabSize() > config.vocab_size) {
        LOG_INFO("Reducing vocabulary to " + std::to_string(config.vocab_size) + " tokens...");
        bpe->reduceVocab(config.vocab_size);
    }

    // Add special tokens if requested
    if (config.add_special_tokens) {
        LOG_INFO("Adding special tokens...");
        bpe->addToken("<EOS>");
        bpe->addToken("<PAD>");
        bpe->addToken("<SOS>");
    }

    LOG_INFO("Final vocabulary size: " + std::to_string(bpe->getVocabSize()));

    // Save checkpoint
    LOG_INFO("Saving checkpoint: " + config.checkpoint_path);
    try {
        bpe->save(config.checkpoint_path);
        LOG_INFO("Training completed successfully!");
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error saving checkpoint: " + std::string(e.what()));
        return -1;
    }

    return 0;
}

int tokenizeMode(const Config& config) {
    MLGL::Timer timer("Tokenization");
    LOG_INFO("Loading BPE model: " + config.checkpoint_path);

    auto bpe = std::make_shared<MLGL::BPE>(1);
    try {
        bpe->load(config.checkpoint_path);
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error loading checkpoint: " + std::string(e.what()));
        return -1;
    }

    std::string text_to_tokenize;

    // Get text to tokenize
    if (!config.input_text.empty()) {
        text_to_tokenize = config.input_text;
    }
    else if (!config.input_files.empty()) {
        std::ifstream file(config.input_files[0]);
        if (!file) {
            LOG_ERROR("Cannot open input file: " + config.input_files[0]);
            return -1;
        }
        std::string line;
        while (std::getline(file, line)) {
            text_to_tokenize += line + "\n";
        }
    }

    // Tokenize
    LOG_DEBUG("Tokenizing " + std::to_string(text_to_tokenize.size()) + " characters...");
    std::vector<std::string> tokens = bpe->tokenizeInput(text_to_tokenize.c_str(), text_to_tokenize.size());

    // Output results
    if (!config.output_file.empty()) {
        std::ofstream out(config.output_file);
        if (!out) {
            LOG_ERROR("Cannot create output file: " + config.output_file);
            return -1;
        }
        for (size_t i = 0; i < tokens.size(); ++i) {
            out << tokens[i];
            if (i < tokens.size() - 1) out << " ";
        }
        out << "\n";
        LOG_INFO("Tokenization results saved to: " + config.output_file);
    }
    else {
        std::cout << "Tokens (" << tokens.size() << "):\n";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << "'" << tokens[i] << "'";
            if (i < tokens.size() - 1) std::cout << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

int infoMode(const Config& config) {
    LOG_INFO("Loading BPE model: " + config.checkpoint_path);

    auto bpe = std::make_shared<MLGL::BPE>(1);
    try {
        bpe->load(config.checkpoint_path);
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error loading checkpoint: " + std::string(e.what()));
        return -1;
    }

    std::cout << "\nBPE Model Information:\n";
    std::cout << "Checkpoint file: " << config.checkpoint_path << "\n";
    std::cout << "Vocabulary size: " << bpe->getVocabSize() << "\n";

    // Additional info if available
    std::filesystem::path path(config.checkpoint_path);
    if (std::filesystem::exists(path)) {
        auto file_size = std::filesystem::file_size(path);
        std::cout << "File size: " << file_size << " bytes\n";

        auto ftime = std::filesystem::last_write_time(path);
        std::cout << "Last modified: " << std::chrono::duration_cast<std::chrono::seconds>(
            ftime.time_since_epoch()).count() << " (seconds since epoch)\n";
    }

    return 0;
}

int reduceMode(const Config& config) {
    MLGL::Timer timer("Vocabulary reduction");
    LOG_INFO("Loading BPE model: " + config.checkpoint_path);

    auto bpe = std::make_shared<MLGL::BPE>(1);
    try {
        bpe->load(config.checkpoint_path);
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error loading checkpoint: " + std::string(e.what()));
        return -1;
    }

    size_t original_vocab_size = bpe->getVocabSize();
    LOG_INFO("Original vocabulary size: " + std::to_string(original_vocab_size));

    if (config.vocab_size >= original_vocab_size) {
        LOG_WARN("Target vocabulary size (" + std::to_string(config.vocab_size) +
            ") is not smaller than current size (" + std::to_string(original_vocab_size) + ")");
        LOG_INFO("No reduction needed.");
        return 0;
    }

    LOG_INFO("Reducing vocabulary from " + std::to_string(original_vocab_size) +
        " to " + std::to_string(config.vocab_size) + " tokens...");

    try {
        bpe->reduceVocab(config.vocab_size);
        LOG_INFO("Vocabulary reduced successfully to " + std::to_string(bpe->getVocabSize()) + " tokens");
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error during vocabulary reduction: " + std::string(e.what()));
        return -1;
    }

    // Determine output path
    std::string output_path = config.output_checkpoint.empty() ? config.checkpoint_path : config.output_checkpoint;

    if (output_path == config.checkpoint_path) {
        LOG_WARN("Overwriting original checkpoint file");
    }
    else {
        LOG_INFO("Saving reduced model to: " + output_path);
    }

    // Save the reduced model
    try {
        bpe->save(output_path);
        LOG_INFO("Reduced model saved successfully!");

        if (config.verbose) {
            std::filesystem::path path(output_path);
            if (std::filesystem::exists(path)) {
                auto file_size = std::filesystem::file_size(path);
                LOG_DEBUG("Output file size: " + std::to_string(file_size) + " bytes");
            }
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error saving reduced model: " + std::string(e.what()));
        return -1;
    }

    return 0;
}

int main(int argc, char** argv) {
    srand(time(nullptr));

    Config config;

    // Parse command line arguments
    if (!parseArguments(argc, argv, config)) {
        return -1;
    }

    // Show help if requested
    if (config.help) {
        printUsage(argv[0]);
        return 0;
    }

    // Validate configuration
    if (!validateConfig(config)) {
        std::cout << "\nUse --help for usage information.\n";
        return -1;
    }

    // Set up logging
    MLGL::Logger::getInstance().setLogLevel(config.verbose ? MLGL::LogLevel::LL_DEBUG : MLGL::LogLevel::LL_INFO);
    MLGL::Logger::getInstance().setEnabled(true);

    // Initialize OpenGL (if needed for compute operations)
    if (!initializeOpenGL()) {
        return -1;
    }

    if (config.verbose) {
        LOG_DEBUG("OpenGL Version: " + std::string(reinterpret_cast<const char*>(glGetString(GL_VERSION))));
    }

    int result = 0;

    // Execute the requested mode
    try {
        if (config.mode == "train") {
            result = trainMode(config);
        }
        else if (config.mode == "tokenize") {
            result = tokenizeMode(config);
        }
        else if (config.mode == "info") {
            result = infoMode(config);
        }
        else if (config.mode == "reduce") {
            result = reduceMode(config);
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("Unhandled exception: " + std::string(e.what()));
        result = -1;
    }

    // Cleanup
    cleanupOpenGL();

    return result;
}