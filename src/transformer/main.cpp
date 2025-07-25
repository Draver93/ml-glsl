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

    {
        std::srand(42);
        std::cout << "=== GPTransformer Training from File (Individual Sentences) ===" << std::endl;

        int d_model = 768;
        int d_hidden = d_model * 2;
        int seq_len = 64;

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
            return -1;
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
            return -1;
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
            return -1;
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

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}