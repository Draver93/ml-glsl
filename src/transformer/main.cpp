#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <random>
#include <csignal>

// Include your headers
#include "GPTransformer.h"
#include "Logger.h"

// GLFW and OpenGL headers
#define GLAD_GLX 0
extern "C" {
#include <glad/glad.h>
}

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// Signal handler for graceful shutdown
volatile bool g_shutdown_requested = false;

struct Config {
    std::string mode = "train";              // train, generate, info, evaluate, interactive
    std::string model_path;                  // Path to model checkpoint
    std::string bpe_path;                    // Path to BPE checkpoint
    std::vector<std::string> input_files;    // Input text files for training
    std::string input_text;                  // Direct text input for generation
    std::string output_file;                 // Output file for results
    std::string prompt;                      // Generation prompt

    // Model architecture parameters
    int d_model = 256;                       // Model dimension
    int d_hidden = 512;                      // Hidden dimension (usually 2x d_model)
    int seq_len = 64;                        // Sequence length

    // Training parameters
    int epochs = 1000;                       // Number of training epochs
    float learning_rate = 0.0001f;           // Initial learning rate
    float lr_decay = 0.95f;                  // Learning rate decay factor
    int lr_decay_steps = 50;                 // Steps between LR decay
    int progress_interval = 1000;              // Progress reporting interval
    int eval_interval = 100;                 // Evaluation interval
    int early_stopping = 500;                // Early stopping patience
    float target_loss = 0.001f;              // Target loss for early stopping

    // Generation parameters
    int max_tokens = 100;                    // Maximum tokens to generate
    float temperature = 1.0f;                // Sampling temperature
    int top_k = 50;                          // Top-K sampling
    bool use_eos = true;                     // Stop at EOS token

    // Evaluation parameters
    int num_eval_prompts = 10;               // Number of evaluation prompts
    bool show_loss_trend = false;            // Show loss trend analysis

    bool verbose = false;                    // Verbose output
    bool help = false;                       // Show help
};

void printUsage(const char* program_name) {
    std::cout << "GPTransformer CLI - Train and run transformer language models\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";

    std::cout << "Modes:\n";
    std::cout << "  --mode <train|generate|info|evaluate|interactive>  Operation mode (default: train)\n\n";

    std::cout << "Common Options:\n";
    std::cout << "  --model <path>                Path to model checkpoint file\n";
    std::cout << "  --bpe <path>                  Path to BPE tokenizer checkpoint\n";
    std::cout << "  --help                        Show this help message\n";
    std::cout << "  --verbose                     Enable verbose output\n\n";

    std::cout << "Model Architecture (for training new models):\n";
    std::cout << "  --d-model <size>              Model dimension (default: 256)\n";
    std::cout << "  --d-hidden <size>             Hidden dimension (default: 512)\n";
    std::cout << "  --seq-len <length>            Sequence length (default: 64)\n\n";

    std::cout << "Training Mode Options:\n";
    std::cout << "  --input <file1,file2,...>     Input text files (comma-separated)\n";
    std::cout << "  --epochs <num>                Number of training epochs (default: 1000)\n";
    std::cout << "  --lr <rate>                   Initial learning rate (default: 0.0001)\n";
    std::cout << "  --lr-decay <factor>           Learning rate decay factor (default: 0.95)\n";
    std::cout << "  --lr-decay-steps <steps>      Steps between LR decay (default: 50)\n";
    std::cout << "  --progress-interval <steps>   Progress reporting interval (default: 1000)\n";
    std::cout << "  --eval-interval <steps>       Evaluation interval (default: 100)\n";
    std::cout << "  --early-stopping <patience>   Early stopping patience (default: 500)\n";
    std::cout << "  --target-loss <loss>          Target loss for early stopping (default: 0.001)\n";
    std::cout << "  --output <file>               Save training log to file\n\n";

    std::cout << "Generation Mode Options:\n";
    std::cout << "  --prompt <text>               Generation prompt (use quotes)\n";
    std::cout << "  --input <file>                Read prompts from file (one per line)\n";
    std::cout << "  --output <file>               Save generated text to file\n";
    std::cout << "  --max-tokens <num>            Maximum tokens to generate (default: 100)\n";
    std::cout << "  --temperature <temp>          Sampling temperature (default: 1.0)\n";
    std::cout << "  --top-k <k>                   Top-K sampling (default: 50)\n";
    std::cout << "  --no-eos                      Don't stop at EOS token\n\n";

    std::cout << "Evaluation Mode Options:\n";
    std::cout << "  --input <file>                Text file for evaluation\n";
    std::cout << "  --num-prompts <num>           Number of evaluation prompts (default: 10)\n";
    std::cout << "  --show-loss-trend             Show loss trend analysis\n\n";

    std::cout << "Interactive Mode Options:\n";
    std::cout << "  --max-tokens <num>            Maximum tokens per response (default: 100)\n";
    std::cout << "  --temperature <temp>          Sampling temperature (default: 1.0)\n\n";

    std::cout << "Examples:\n";
    std::cout << "  # Train new model\n";
    std::cout << "  " << program_name << " --mode train --bpe tokenizer.bpe --model model.gpt --input data.txt\n\n";
    std::cout << "  # Generate text\n";
    std::cout << "  " << program_name << " --mode generate --model model.gpt --prompt \"Once upon a time\"\n\n";
    std::cout << "  # Interactive chat\n";
    std::cout << "  " << program_name << " --mode interactive --model model.gpt\n\n";
    std::cout << "  # Evaluate model\n";
    std::cout << "  " << program_name << " --mode evaluate --model model.gpt --input test_data.txt\n\n";
    std::cout << "  # Model info\n";
    std::cout << "  " << program_name << " --mode info --model model.gpt\n";
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
            if (config.mode != "train" && config.mode != "generate" &&
                config.mode != "info" && config.mode != "evaluate" &&
                config.mode != "interactive") {
                std::cerr << "Error: Invalid mode '" << config.mode << "'\n";
                return false;
            }
        }
        else if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        }
        else if (arg == "--bpe" && i + 1 < argc) {
            config.bpe_path = argv[++i];
        }
        else if (arg == "--input" && i + 1 < argc) {
            std::string input_str = argv[++i];
            if (config.mode == "train") {
                config.input_files = split(input_str, ',');
            }
            else {
                config.input_files = { input_str };
            }
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        }
        else if (arg == "--prompt" && i + 1 < argc) {
            config.prompt = argv[++i];
        }
        else if (arg == "--d-model" && i + 1 < argc) {
            config.d_model = std::stoi(argv[++i]);
        }
        else if (arg == "--d-hidden" && i + 1 < argc) {
            config.d_hidden = std::stoi(argv[++i]);
        }
        else if (arg == "--seq-len" && i + 1 < argc) {
            config.seq_len = std::stoi(argv[++i]);
        }
        else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::stoi(argv[++i]);
        }
        else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = std::stof(argv[++i]);
        }
        else if (arg == "--lr-decay" && i + 1 < argc) {
            config.lr_decay = std::stof(argv[++i]);
        }
        else if (arg == "--lr-decay-steps" && i + 1 < argc) {
            config.lr_decay_steps = std::stoi(argv[++i]);
        }
        else if (arg == "--progress-interval" && i + 1 < argc) {
            config.progress_interval = std::stoi(argv[++i]);
        }
        else if (arg == "--eval-interval" && i + 1 < argc) {
            config.eval_interval = std::stoi(argv[++i]);
        }
        else if (arg == "--early-stopping" && i + 1 < argc) {
            config.early_stopping = std::stoi(argv[++i]);
        }
        else if (arg == "--target-loss" && i + 1 < argc) {
            config.target_loss = std::stof(argv[++i]);
        }
        else if (arg == "--max-tokens" && i + 1 < argc) {
            config.max_tokens = std::stoi(argv[++i]);
        }
        else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        }
        else if (arg == "--top-k" && i + 1 < argc) {
            config.top_k = std::stoi(argv[++i]);
        }
        else if (arg == "--num-prompts" && i + 1 < argc) {
            config.num_eval_prompts = std::stoi(argv[++i]);
        }
        else if (arg == "--no-eos") {
            config.use_eos = false;
        }
        else if (arg == "--show-loss-trend") {
            config.show_loss_trend = true;
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
    if (config.mode == "train") {
        if (config.model_path.empty()) {
            std::cerr << "Error: Model path required for training (--model)\n";
            return false;
        }
        if (config.bpe_path.empty() && !std::filesystem::exists(config.model_path)) {
            std::cerr << "Error: BPE tokenizer path required for initial training (--bpe)\n";
            return false;
        }
        if (!config.bpe_path.empty() && std::filesystem::exists(config.model_path)) {
            std::cerr << "Error: BPE tokenizer path required only for initial training (--bpe)\n";
            return false;
        }
        if (config.input_files.empty()) {
            std::cerr << "Error: Input files required for training (--input)\n";
            return false;
        }
        for (const auto& file : config.input_files) {
            if (!std::filesystem::exists(file)) {
                std::cerr << "Error: Input file '" << file << "' does not exist\n";
                return false;
            }
        }
        if (!config.bpe_path.empty() && !std::filesystem::exists(config.bpe_path)) {
            std::cerr << "Error: BPE file '" << config.bpe_path << "' does not exist\n";
            return false;
        }
    }
    else if (config.mode == "generate" || config.mode == "interactive" ||
        config.mode == "evaluate" || config.mode == "info") {
        if (config.model_path.empty()) {
            std::cerr << "Error: Model path required (--model)\n";
            return false;
        }
        if (!std::filesystem::exists(config.model_path)) {
            std::cerr << "Error: Model file '" << config.model_path << "' does not exist\n";
            return false;
        }

        if (config.mode == "generate" && config.prompt.empty() && config.input_files.empty()) {
            std::cerr << "Error: Either --prompt or --input required for generation\n";
            return false;
        }
        if (config.mode == "evaluate" && config.input_files.empty()) {
            std::cerr << "Error: Input file required for evaluation (--input)\n";
            return false;
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

    GLFWwindow* window = glfwCreateWindow(1, 1, "GPT Compute", nullptr, nullptr);
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

std::vector<std::string> loadTextData(const std::vector<std::string>& filenames) {
    std::vector<std::string> data;

    for (const auto& filename : filenames) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            LOG_ERROR("Could not open file: " + filename);
            continue;
        }

        std::string line;
        int line_count = 0;
        while (std::getline(file, line)) {
            line_count++;
            if (line.empty() || line.length() < 3) continue;

            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            if (!line.empty()) {
                data.push_back(line);
            }
        }

        LOG_INFO("Loaded " + std::to_string(data.size()) + " lines from " + filename);
    }

    return data;
}

int trainMode(const Config& config) {
    LOG_INFO("=== GPTransformer Training Mode ===");

    // Load training data
    std::vector<std::string> training_data = loadTextData(config.input_files);
    if (training_data.empty()) {
        LOG_ERROR("No training data loaded!");
        return -1;
    }

    LOG_INFO("Loaded " + std::to_string(training_data.size()) + " training examples");

    // Create or load model
    std::shared_ptr<MLGL::GPTransformer> gpt;
    if (std::filesystem::exists(config.model_path)) {
        LOG_INFO("Loading existing model: " + config.model_path);
        gpt = std::make_shared<MLGL::GPTransformer>(config.model_path);
    }
    else {
        LOG_INFO("Creating new model with BPE: " + config.bpe_path);
        gpt = std::make_shared<MLGL::GPTransformer>(
            config.bpe_path, config.d_model, config.d_hidden, config.seq_len);
    }

    // Prepare training examples (chunk sentences into windows matching seq_len)
    struct TrainingExample {
        std::vector<std::string> tokens;   // includes <SOS> ... <EOS>
        std::string original_text;
    };

    std::vector<TrainingExample> examples;

    // Build a continuous token stream across lines and pack exact-length examples (no padding).
    const int max_inner = std::max(1, config.seq_len - 2);
    std::vector<std::string> token_buffer;
    token_buffer.reserve(16 * static_cast<size_t>(max_inner));
    size_t cursor = 0;

    for (const auto& sentence : training_data) {
        std::vector<std::string> sentence_tokens = gpt->tokenizeInput(sentence.c_str(), sentence.size());
        if (!sentence_tokens.empty()) {
            token_buffer.insert(token_buffer.end(), sentence_tokens.begin(), sentence_tokens.end());
        }

        // While we can fill one full example, emit it
        while (cursor + static_cast<size_t>(max_inner) <= token_buffer.size()) {
            TrainingExample ex;
            ex.original_text = sentence;
            ex.tokens.clear();
            ex.tokens.push_back("<SOS>");
            ex.tokens.insert(ex.tokens.end(), token_buffer.begin() + static_cast<std::ptrdiff_t>(cursor),
                             token_buffer.begin() + static_cast<std::ptrdiff_t>(cursor + static_cast<size_t>(max_inner)));
            ex.tokens.push_back("<EOS>");
            if (ex.tokens.size() == static_cast<size_t>(config.seq_len)) {
                examples.push_back(std::move(ex));
            }
            cursor += static_cast<size_t>(max_inner);

            // Periodically compact buffer to avoid growth
            if (cursor > token_buffer.size() / 2) {
                token_buffer.erase(token_buffer.begin(), token_buffer.begin() + static_cast<std::ptrdiff_t>(cursor));
                cursor = 0;
            }
        }
    }

    LOG_INFO("Prepared " + std::to_string(examples.size()) + " training examples");

    // Initialize random number generator
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // Keep track of last 10 trained examples for evaluation
    std::deque<size_t> last_trained_examples;
    const size_t MAX_LAST_EXAMPLES = 10;

    // Function to get random evaluation prompt from last 10 examples
    auto getRandomEvalPrompt = [&]() -> std::pair<std::string, std::string> {
        if (last_trained_examples.empty()) {
            return { "", "" };
        }

        // Pick random example from last 10
        std::uniform_int_distribution<size_t> dist(0, last_trained_examples.size() - 1);
        size_t random_idx = dist(gen);
        size_t example_idx = last_trained_examples[random_idx];

        const auto& example = examples[example_idx];
        if (example.tokens.size() > 2) {
            std::string prompt, display;
            size_t prefix_len = std::max(1, (int)(example.tokens.size() - 1) / 2);
            for (size_t j = 0; j < prefix_len; ++j) {
                prompt += example.tokens[j];
                display += "'" + example.tokens[j] + "' ";
            }
            return { display, prompt };
        }
        return { "", "" };
    };

    // Training loop
    LOG_INFO("Starting training...");
    float best_loss = std::numeric_limits<float>::infinity();
    int epochs_without_improvement = 0;
    std::vector<float> epoch_losses;
    int total_token_trained = 0;

    // Keep track of trained examples
    std::vector<size_t> trained_indices;

    std::ofstream log_file;
    if (!config.output_file.empty()) {
        log_file.open(config.output_file);
        log_file << "epoch,line,loss,lr,tokens_trained\n";
    }

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        float learning_rate = config.learning_rate *
            std::pow(config.lr_decay, epoch / (float)config.lr_decay_steps);

        int total_predictions = 0;

        // Shuffle examples
        std::vector<size_t> indices(examples.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        for (size_t idx : indices) {
            const auto& example = examples[idx];

            for (size_t i = 1; i < example.tokens.size(); ++i) {
                std::vector<std::string> context(example.tokens.begin(), example.tokens.begin() + i);
                std::string target = example.tokens[i];
                gpt->trainNextToken(context, target, learning_rate);
                total_token_trained++;

                if (total_token_trained % config.progress_interval == 0) {
                    float avg_loss = gpt->getAvrLoss();
                    std::cout << "Tokens: " << total_token_trained
                        << " | Epoch: " << (epoch + 1)
                        << " | Avg Loss: " << std::fixed << std::setprecision(4) << avg_loss
                        << " | LR: " << std::scientific << std::setprecision(2) << learning_rate
                        << std::endl;

                    if (log_file.is_open()) {
                        log_file << epoch << "," << total_token_trained << ","
                            << avg_loss << "," << learning_rate << ","
                            << total_predictions << "\n";
                    }

                    // Get random evaluation prompt from last 10 examples
                    auto eval_prompt = getRandomEvalPrompt();
                    if (!eval_prompt.first.empty()) {
                        std::string prediction = gpt->eval(eval_prompt.second);
                        std::cout << "  Sample: [" << eval_prompt.first << "] -> '"
                            << prediction << "'\n";
                    }
                }
            }

            // Add to last trained examples (keep only last 10)
            last_trained_examples.push_back(idx);
            if (last_trained_examples.size() > MAX_LAST_EXAMPLES) {
                last_trained_examples.pop_front();
            }

            if (g_shutdown_requested) break;
        }

        float avg_loss = gpt->getAvrLoss();
        epoch_losses.push_back(avg_loss);

        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            epochs_without_improvement = 0;
            // Save best model
            gpt->save(config.model_path);
            LOG_DEBUG("Saved improved model (loss: " + std::to_string(best_loss) + ")");
        }
        else {
            epochs_without_improvement++;
        }

        // Detailed evaluation with random prompts from last 10 examples
        if ((epoch + 1) % config.eval_interval == 0) {
            std::cout << "\n=== Epoch " << (epoch + 1) << " Evaluation ===\n";
            std::cout << "Avg Loss: " << std::fixed << std::setprecision(4) << avg_loss
                << " | Best: " << best_loss
                << " | No Improve: " << epochs_without_improvement << "\n";

            // Show 3 different random evaluations from last 10 examples
            for (int i = 0; i < 3; ++i) {
                auto eval_prompt = getRandomEvalPrompt();
                if (!eval_prompt.first.empty()) {
                    std::string prediction = gpt->eval(eval_prompt.second);
                    std::cout << "  [" << eval_prompt.first << "] -> '" << prediction << "'\n";
                }
            }
            std::cout << std::endl;
        }

        // Early stopping checks
        if (epochs_without_improvement > config.early_stopping) {
            LOG_INFO("Early stopping - no improvement for " +
                std::to_string(config.early_stopping) + " epochs");
            break;
        }
        if (g_shutdown_requested) break;

        
        if (avg_loss < config.target_loss) {
            LOG_INFO("Target loss achieved: " + std::to_string(avg_loss));
            break;
        }
    }

    // Final save
    gpt->save(config.model_path);
    LOG_INFO("Training completed. Model saved to: " + config.model_path);
    LOG_INFO("Best loss achieved: " + std::to_string(best_loss));

    return 0;
}

int generateMode(const Config& config) {
    LOG_INFO("=== GPTransformer Generation Mode ===");

    // Load model
    auto gpt = std::make_shared<MLGL::GPTransformer>(config.model_path);
    LOG_INFO("Model loaded from: " + config.model_path);

    std::vector<std::string> prompts;

    if (!config.prompt.empty()) {
        prompts.push_back(config.prompt);
    }
    else if (!config.input_files.empty()) {
        std::ifstream file(config.input_files[0]);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                prompts.push_back(line);
            }
        }
    }

    std::ostream* output = &std::cout;
    std::ofstream out_file;
    if (!config.output_file.empty()) {
        out_file.open(config.output_file);
        output = &out_file;
    }

    for (size_t i = 0; i < prompts.size(); ++i) {
        const std::string& prompt = prompts[i];
        std::cout << "Generating for prompt " << (i + 1) << "/" << prompts.size()
            << ": \"" << prompt << "\"\n";

        std::string generated = gpt->eval(prompt);

        *output << "Prompt: " << prompt << "\n";
        *output << "Generated: " << generated << "\n";
        if (prompts.size() > 1) *output << "---\n";

        std::cout << "Generated: " << generated << "\n\n";
    }

    return 0;
}

int interactiveMode(const Config& config) {
    LOG_INFO("=== GPTransformer Interactive Mode ===");

    auto gpt = std::make_shared<MLGL::GPTransformer>(config.model_path);
    LOG_INFO("Model loaded. Type 'quit' or 'exit' to end session.\n");

    std::string input;
    while (true) {
        if (g_shutdown_requested) break;
        std::cout << "> ";
        std::getline(std::cin, input);

        if (input == "quit" || input == "exit" || input == "q") {
            break;
        }

        if (input.empty()) {
            continue;
        }
        input = "<SOS>" + input;
        std::string response = gpt->eval(input);
        std::cout << response << "\n\n";

    }

    std::cout << "Interactive session ended.\n";
    return 0;
}

int evaluateMode(const Config& config) {
    LOG_INFO("=== GPTransformer Evaluation Mode ===");

    auto gpt = std::make_shared<MLGL::GPTransformer>(config.model_path);
    std::vector<std::string> test_data = loadTextData(config.input_files);

    if (test_data.empty()) {
        LOG_ERROR("No evaluation data loaded!");
        return -1;
    }

    LOG_INFO("Evaluating on " + std::to_string(test_data.size()) + " examples");

    // Create test prompts (first half of each sentence)
    std::vector<std::pair<std::string, std::string>> test_cases;
    int max_tests = std::min(config.num_eval_prompts, (int)test_data.size());

    for (int i = 0; i < max_tests; ++i) {
        const std::string& sentence = test_data[i];
        if (sentence.length() > 10) {
            size_t split_pos = sentence.length() / 2;
            std::string prompt = sentence.substr(0, split_pos);
            std::string expected = sentence.substr(split_pos);
            test_cases.emplace_back(prompt, expected);
        }
    }

    std::cout << "\n=== Evaluation Results ===\n";
    for (size_t i = 0; i < test_cases.size(); ++i) {
        const auto& [prompt, expected] = test_cases[i];
        std::string generated = gpt->eval(prompt);

        std::cout << "Test " << (i + 1) << ":\n";
        std::cout << "  Prompt: \"" << prompt << "\"\n";
        std::cout << "  Expected: \"" << expected << "\"\n";
        std::cout << "  Generated: \"" << generated << "\"\n";
        std::cout << "  Match: " << (generated.find(expected.substr(0, 10)) != std::string::npos ? "Partial" : "No") << "\n\n";
    }

    return 0;
}

int infoMode(const Config& config) {
    LOG_INFO("=== GPTransformer Model Information ===");

    try {
        auto gpt = std::make_shared<MLGL::GPTransformer>(config.model_path);

        std::cout << "\nModel Information:\n";
        std::cout << "Model file: " << config.model_path << "\n";

        std::filesystem::path path(config.model_path);
        if (std::filesystem::exists(path)) {
            auto file_size = std::filesystem::file_size(path);
            std::cout << "File size: " << file_size << " bytes ("
                << std::fixed << std::setprecision(2) << file_size / (1024.0 * 1024.0) << " MB)\n";

            auto ftime = std::filesystem::last_write_time(path);
            auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
            std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);
            std::cout << "Last modified: " << std::put_time(std::localtime(&cftime), "%Y-%m-%d %H:%M:%S") << "\n";
        }

        // Test model with a simple prompt
        std::cout << "\nModel Test:\n";
        std::string test_prompt = "The quick brown";
        std::string test_output = gpt->eval(test_prompt);
        std::cout << "Test prompt: \"" << test_prompt << "\"\n";
        std::cout << "Model output: \"" << test_output << "\"\n";

        std::cout << "\nModel appears to be working correctly.\n";

    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return -1;
    }

    return 0;
}

// Performance benchmarking function
void benchmarkModel(const Config& config) {
    if (config.model_path.empty()) {
        std::cerr << "Error: Model path required for benchmarking\n";
        return;
    }

    try {
        LOG_INFO("=== Model Performance Benchmark ===");
        auto gpt = std::make_shared<MLGL::GPTransformer>(config.model_path);

        std::vector<std::string> test_prompts = {
            "The weather today is",
            "In the beginning",
            "Machine learning is",
            "The future of technology",
            "Once upon a time"
        };

        std::cout << "\nBenchmarking generation speed...\n";
        auto start_time = std::chrono::high_resolution_clock::now();

        for (const auto& prompt : test_prompts) {
            auto prompt_start = std::chrono::high_resolution_clock::now();
            std::string result = gpt->eval(prompt);
            auto prompt_end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(prompt_end - prompt_start);
            std::cout << "Prompt: \"" << prompt << "\" -> \"" << result << "\" ("
                << duration.count() << "ms)\n";
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "\nTotal time: " << total_duration.count() << "ms\n";
        std::cout << "Average per prompt: " << total_duration.count() / test_prompts.size() << "ms\n";

    }
    catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << "\n";
    }
}

// Enhanced generation with sampling parameters
std::string generateWithSampling(MLGL::GPTransformer* gpt, const std::string& prompt,
    int max_tokens, float temperature, int top_k, bool use_eos) {
    // This would need to be implemented in the GPTransformer class
    // For now, using the basic eval function
    return gpt->eval(prompt);
}

// Loss trend analysis for training
void analyzeLossTrend(const std::vector<float>& losses) {
    if (losses.size() < 10) {
        std::cout << "Not enough data points for trend analysis\n";
        return;
    }

    std::cout << "\n=== Loss Trend Analysis ===\n";

    // Calculate moving average
    std::vector<float> moving_avg;
    int window = std::min(10, (int)losses.size() / 4);

    for (size_t i = window; i < losses.size(); ++i) {
        float sum = 0;
        for (int j = 0; j < window; ++j) {
            sum += losses[i - j];
        }
        moving_avg.push_back(sum / window);
    }

    // Find trend
    float slope = 0;
    if (moving_avg.size() > 1) {
        slope = (moving_avg.back() - moving_avg.front()) / (moving_avg.size() - 1);
    }

    std::cout << "Final loss: " << std::fixed << std::setprecision(6) << losses.back() << "\n";
    std::cout << "Best loss: " << *std::min_element(losses.begin(), losses.end()) << "\n";
    std::cout << "Trend slope: " << slope << " (";
    if (slope < -0.001) std::cout << "Improving";
    else if (slope > 0.001) std::cout << "Worsening";
    else std::cout << "Stable";
    std::cout << ")\n";

    // Show recent performance
    if (losses.size() >= 20) {
        auto recent_start = losses.end() - 10;
        float recent_avg = std::accumulate(recent_start, losses.end(), 0.0f) / 10.0f;
        std::cout << "Recent 10-epoch average: " << recent_avg << "\n";
    }
}

void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_shutdown_requested = true;
        std::cout << "\nShutdown requested. Saving progress...\n";
    }
}



// Main function
int main(int argc, char** argv) {
    // Simulated command line arguments
#ifdef DEBUG
    std::vector<const char*> args = {
        "ml-glsl-transformer.exe",
        "--bpe", "tokens.bpe",
        "--mode", "train",
        "--verbose",
        "--seq-len", "1024",
        "--progress-interval", "500",
        "--model", "model.gpt",
        "--input", "pg51161.txt"
    };

    int argc_mock = args.size();
    char** argv_mock = new char* [argc_mock];

    for (int i = 0; i < argc_mock; ++i) {
        size_t len = std::strlen(args[i]);
        argv_mock[i] = new char[len + 1];       // Allocate space for null-terminator
        std::strcpy(argv_mock[i], args[i]);     // Copy the C-string into argv
    }
#endif // DEBUG


    // Set up signal handling
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    Config config;

    // Parse command line arguments
#ifdef DEBUG
    if (!parseArguments(argc_mock, argv_mock, config)) { printUsage(argv_mock[0]); return -1; }
#else
    if (!parseArguments(argc, argv, config)) { printUsage(argv[0]); return -1; }
#endif // DEBUG

    if (config.help) {
        printUsage(argv[0]);
        return 0;
    }

    // Validate configuration
    if (!validateConfig(config)) {
        return -1;
    }

    // Initialize OpenGL context
    if (!initializeOpenGL()) {
        LOG_ERROR("Failed to initialize OpenGL context");
        return -1;
    }

    // Set logging level
    if (config.verbose) {
        // Assuming Logger has a setLevel function
        LOG_INFO("Verbose mode enabled");
    }

    // Set up logging
    MLGL::Logger::getInstance().setLogLevel(config.verbose ? MLGL::LogLevel::LL_DEBUG : MLGL::LogLevel::LL_INFO);
    MLGL::Logger::getInstance().setEnabled(true);

    int result = 0;

    try {
        // Execute the appropriate mode
        if (config.mode == "train") {
            result = trainMode(config);
        }
        else if (config.mode == "generate") {
            result = generateMode(config);
        }
        else if (config.mode == "interactive") {
            result = interactiveMode(config);
        }
        else if (config.mode == "evaluate") {
            result = evaluateMode(config);
        }
        else if (config.mode == "info") {
            result = infoMode(config);
        }
        else if (config.mode == "benchmark") {
            benchmarkModel(config);
            result = 0;
        }
        else {
            std::cerr << "Error: Unknown mode '" << config.mode << "'\n";
            result = -1;
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception occurred: " + std::string(e.what()));
        result = -1;
    }
    catch (...) {
        LOG_ERROR("Unknown exception occurred");
        result = -1;
    }

    // Cleanup
    cleanupOpenGL();

    if (result == 0) {
        LOG_INFO("Program completed successfully");
    }
    else {
        LOG_ERROR("Program completed with errors");
    }

    return result;
}