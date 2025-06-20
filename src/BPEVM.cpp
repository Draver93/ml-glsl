#include "BPEVM.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <filesystem>

namespace NNGL {

    template <typename T1, typename T2>
    std::size_t PairHash::operator()(const std::pair<T1, T2>& p) const noexcept {
        size_t h1 = std::hash<T1>{}(p.first);
        size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }

    size_t VectorStringHash::operator()(const std::vector<std::string>& vec) const noexcept {
        size_t seed = 0;
        for (const auto& s : vec) {
            seed ^= std::hash<std::string>{}(s)+0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }

    bool VectorStringEqual::operator()(const std::vector<std::string>& lhs, const std::vector<std::string>& rhs) const noexcept {
        return lhs == rhs;
    }

    void BytePairEncodingVectorMapper::clean_word(std::string& word) noexcept {
        // Currently disabled - enable if punctuation removal is needed
        return;
        word.erase(std::remove_if(word.begin(), word.end(),
            [](unsigned char c) { return std::ispunct(c); }), word.end());
        std::transform(word.begin(), word.end(), word.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    }

    void BytePairEncodingVectorMapper::build_vocab(const std::vector<std::string>& corpus) {
        vocab_.clear();
        base_tokens_.clear();

        for (auto word : corpus) {
            clean_word(word);
            if (word.empty()) continue;

            std::vector<std::string> tokens;
            tokens.reserve(word.size());
            for (char c : word) {
                std::string char_token(1, c);
                tokens.push_back(char_token);
                base_tokens_.insert(char_token);
            }
            ++vocab_[std::move(tokens)];
        }

        initializeVectorsForNewTokens();
    }

    void BytePairEncodingVectorMapper::update_vocab(const std::vector<std::string>& new_corpus) {
        // Add new words to existing vocabulary
        for (auto word : new_corpus) {
            clean_word(word);
            if (word.empty()) continue;

            std::vector<std::string> tokens;
            tokens.reserve(word.size());
            for (char c : word) {
                std::string char_token(1, c);
                tokens.push_back(char_token);
                base_tokens_.insert(char_token);
            }

            // Apply existing merges to the new word
            for (const auto& merge : merges_) {
                applyMergeToTokens(tokens, merge);
            }

            ++vocab_[std::move(tokens)];
        }

        initializeVectorsForNewTokens();
    }

    void BytePairEncodingVectorMapper::applyMergeToTokens(std::vector<std::string>& tokens, const TokenPair& merge) const {
        std::vector<std::string> new_tokens;
        new_tokens.reserve(tokens.size());

        for (size_t i = 0; i < tokens.size(); ) {
            if (i + 1 < tokens.size() && tokens[i] == merge.first && tokens[i + 1] == merge.second) {
                new_tokens.emplace_back(tokens[i] + tokens[i + 1]);
                i += 2;
            }
            else {
                new_tokens.push_back(tokens[i]);
                ++i;
            }
        }
        tokens = std::move(new_tokens);
    }

    std::unordered_map<BytePairEncodingVectorMapper::TokenPair, int, PairHash>
        BytePairEncodingVectorMapper::get_pair_frequencies() const noexcept {
        std::unordered_map<TokenPair, int, PairHash> freqs;

        for (const auto& [tokens, freq] : vocab_) {
            if (tokens.size() < 2) continue;
            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                freqs[{tokens[i], tokens[i + 1]}] += freq;
            }
        }

        return freqs;
    }

    void BytePairEncodingVectorMapper::merge_pair(const TokenPair& pair_to_merge) {
        std::unordered_map<std::vector<std::string>, int, VectorStringHash, VectorStringEqual> new_vocab;

        for (const auto& [tokens, freq] : vocab_) {
            std::vector<std::string> new_tokens = tokens;
            applyMergeToTokens(new_tokens, pair_to_merge);
            new_vocab[std::move(new_tokens)] += freq;
        }

        vocab_ = std::move(new_vocab);
        updateVectorsAfterMerge(pair_to_merge);
    }

    void BytePairEncodingVectorMapper::performTrainingSteps(const std::string& checkpoint_prefix, int target_merges) {
        int end_merge = (target_merges > 0) ? target_merges : std::numeric_limits<int>::max();

        for (int i = current_merge_step_; i < end_merge; ++i) {
            auto pair_freqs = get_pair_frequencies();
            if (pair_freqs.empty()) {
                std::cout << "No more pairs to merge at step " << i << "\n";
                break;
            }

            auto best_pair_it = std::max_element(pair_freqs.begin(), pair_freqs.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

            // Stop if frequency is too low (configurable threshold)
            if (best_pair_it->second < min_merge_frequency_) {
                break;
            }

            std::cout << "Merge " << i + 1;
            if (target_merges > 0) std::cout << "/" << target_merges;
            std::cout << ": '" << best_pair_it->first.first << "' + '" << best_pair_it->first.second
                << "' (freq: " << best_pair_it->second << ")\n";

            merges_.push_back(best_pair_it->first);
            merge_pair(best_pair_it->first);
            current_merge_step_ = i + 1;

            // Periodic checkpoint saving
            if ((i + 1) % save_interval_ == 0) {
                std::cout << "Saving checkpoint at merge " << (i + 1) << "...\n";
                if (!saveCheckpoint(checkpoint_prefix, i + 1)) {
                    std::cerr << "Warning: Failed to save checkpoint\n";
                }
            }
        }

        is_trained_ = true;
    }

    void BytePairEncodingVectorMapper::train(const std::vector<std::string>& corpus, const std::string& checkpoint_prefix, int target_merges) {
        if (is_trained_) {
            update_vocab(corpus);
        }
        else {
            build_vocab(corpus);
        }

        performTrainingSteps(checkpoint_prefix, target_merges);
    }

    void BytePairEncodingVectorMapper::train(const std::string& filename, const std::string& checkpoint_prefix, int target_merges) {
        std::ifstream input(filename);
        if (!input) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }

        std::vector<std::string> corpus;
        corpus.reserve(corpus_chunk_size_);
        std::string line;

        while (std::getline(input, line)) {
            if (!line.empty()) {
                corpus.push_back(line);
                if (corpus.size() >= corpus_chunk_size_) {
                    std::cout << "Final vocabulary size: " << getAllTokens().size() << " tokens\n";
                    train(corpus, checkpoint_prefix, target_merges);
                    corpus.clear();
                }
            }
        }

        if (!corpus.empty()) {
            std::cout << "Final vocabulary size: " << getAllTokens().size() << " tokens\n";
            train(corpus, checkpoint_prefix, target_merges);
        }

        std::cout << "Training on file completed. Total merges: " << merges_.size() << "\n";
    }

    void BytePairEncodingVectorMapper::resetTraining() {
        vocab_.clear();
        base_tokens_.clear();
        merges_.clear();
        current_merge_step_ = 0;
        is_trained_ = false;

        std::cout << "Training state reset. Existing token vectors preserved.\n";
    }

    bool BytePairEncodingVectorMapper::reduceVocabulary(int max_tokens, VocabReductionStrategy strategy) {
        if (max_tokens <= 0) {
            std::cerr << "Invalid max_tokens value: " << max_tokens << std::endl;
            return false;
        }

        auto all_tokens = getAllTokens();
        if (static_cast<int>(all_tokens.size()) <= max_tokens) {
            std::cout << "Vocabulary already within target size (" << all_tokens.size()
                << " <= " << max_tokens << ")\n";
            return true;
        }

        std::vector<std::string> tokens_to_remove;

        switch (strategy) {
        case VocabReductionStrategy::REMOVE_LEAST_FREQUENT:
            tokens_to_remove = selectTokensByFrequency(max_tokens, false);
            break;
        case VocabReductionStrategy::REMOVE_MOST_RECENT:
            tokens_to_remove = selectTokensByRecency(max_tokens);
            break;
        case VocabReductionStrategy::REMOVE_LONGEST:
            tokens_to_remove = selectTokensByLength(max_tokens, false);
            break;
        case VocabReductionStrategy::REMOVE_SHORTEST:
            tokens_to_remove = selectTokensByLength(max_tokens, true);
            break;
        }

        return removeTokens(tokens_to_remove);
    }

    std::vector<std::string> BytePairEncodingVectorMapper::selectTokensByFrequency(int max_tokens, bool keep_frequent) const {
        std::unordered_map<std::string, int> token_frequencies;

        // Count token frequencies in current vocabulary
        for (const auto& [tokens, freq] : vocab_) {
            for (const auto& token : tokens) {
                token_frequencies[token] += freq;
            }
        }

        // Convert to vector for sorting
        std::vector<std::pair<std::string, int>> freq_pairs;
        for (const auto& token : getAllTokens()) {
            if (base_tokens_.find(token) == base_tokens_.end()) { // Don't remove base tokens
                freq_pairs.emplace_back(token, token_frequencies[token]);
            }
        }

        // Sort by frequency
        std::sort(freq_pairs.begin(), freq_pairs.end(),
            [keep_frequent](const auto& a, const auto& b) {
                return keep_frequent ? (a.second > b.second) : (a.second < b.second);
            });

        // Select tokens to remove
        std::vector<std::string> tokens_to_remove;
        int current_vocab_size = static_cast<int>(getAllTokens().size());
        int tokens_to_remove_count = current_vocab_size - max_tokens;

        for (int i = 0; i < std::min(tokens_to_remove_count, static_cast<int>(freq_pairs.size())); ++i) {
            tokens_to_remove.push_back(freq_pairs[i].first);
        }

        return tokens_to_remove;
    }

    std::vector<std::string> BytePairEncodingVectorMapper::selectTokensByRecency(int max_tokens) const {
        std::vector<std::string> tokens_to_remove;
        auto all_tokens = getAllTokens();
        int current_vocab_size = static_cast<int>(all_tokens.size());
        int tokens_to_remove_count = current_vocab_size - max_tokens;

        // Remove most recently created merged tokens (from end of merges list)
        for (int i = static_cast<int>(merges_.size()) - 1;
            i >= 0 && static_cast<int>(tokens_to_remove.size()) < tokens_to_remove_count;
            --i) {
            std::string merged_token = merges_[i].first + merges_[i].second;
            tokens_to_remove.push_back(merged_token);
        }

        return tokens_to_remove;
    }

    std::vector<std::string> BytePairEncodingVectorMapper::selectTokensByLength(int max_tokens, bool remove_short) const {
        std::vector<std::string> non_base_tokens;

        for (const auto& token : getAllTokens()) {
            if (base_tokens_.find(token) == base_tokens_.end()) {
                non_base_tokens.push_back(token);
            }
        }

        // Sort by length
        std::sort(non_base_tokens.begin(), non_base_tokens.end(),
            [remove_short](const std::string& a, const std::string& b) {
                return remove_short ? (a.length() < b.length()) : (a.length() > b.length());
            });

        // Select tokens to remove
        std::vector<std::string> tokens_to_remove;
        int current_vocab_size = static_cast<int>(getAllTokens().size());
        int tokens_to_remove_count = current_vocab_size - max_tokens;

        for (int i = 0; i < std::min(tokens_to_remove_count, static_cast<int>(non_base_tokens.size())); ++i) {
            tokens_to_remove.push_back(non_base_tokens[i]);
        }

        return tokens_to_remove;
    }

    bool BytePairEncodingVectorMapper::removeTokens(const std::vector<std::string>& tokens_to_remove) {
        if (tokens_to_remove.empty()) return true;

        std::unordered_set<std::string> remove_set(tokens_to_remove.begin(), tokens_to_remove.end());

        // Remove from token vectors
        for (const auto& token : tokens_to_remove) {
            token_vectors_.erase(token);
        }

        // Remove corresponding merges and rebuild vocabulary
        std::vector<TokenPair> new_merges;
        for (const auto& merge : merges_) {
            std::string merged_token = merge.first + merge.second;
            if (remove_set.find(merged_token) == remove_set.end()) {
                new_merges.push_back(merge);
            }
        }

        merges_ = std::move(new_merges);
        current_merge_step_ = static_cast<int>(merges_.size());

        // Rebuild vocabulary with remaining merges
        rebuildVocabulary();

        std::cout << "Removed " << tokens_to_remove.size() << " tokens. "
            << "New vocabulary size: " << getAllTokens().size() << std::endl;

        return true;
    }

    void BytePairEncodingVectorMapper::rebuildVocabulary() {
        // Store original word frequencies
        std::unordered_map<std::string, int> word_frequencies;
        for (const auto& [tokens, freq] : vocab_) {
            std::string word;
            for (const auto& token : tokens) {
                word += token;
            }
            word_frequencies[word] += freq;
        }

        // Rebuild vocabulary from scratch
        vocab_.clear();

        for (const auto& [word, freq] : word_frequencies) {
            std::vector<std::string> tokens;
            tokens.reserve(word.size());

            for (char c : word) {
                tokens.emplace_back(1, c);
            }

            // Apply current merges
            for (const auto& merge : merges_) {
                applyMergeToTokens(tokens, merge);
            }

            vocab_[std::move(tokens)] += freq;
        }
    }

    std::vector<std::string> BytePairEncodingVectorMapper::encode(const std::string& word) const noexcept {
        std::vector<std::string> tokens;
        tokens.reserve(word.size());

        for (char c : word) {
            tokens.emplace_back(1, c);
        }

        for (const auto& merge : merges_) {
            applyMergeToTokens(tokens, merge);
        }

        return tokens;
    }

    std::set<std::string> BytePairEncodingVectorMapper::getAllTokens() const {
        std::set<std::string> all_tokens;

        all_tokens.insert(base_tokens_.begin(), base_tokens_.end());

        for (const auto& [first, second] : merges_) {
            all_tokens.insert(first + second);
        }

        return all_tokens;
    }

    void BytePairEncodingVectorMapper::addToken(const std::string& token) {
        if (token_vectors_.find(token) == token_vectors_.end()) {
            token_vectors_[token] = initRandomVector();
        }
    }

    void BytePairEncodingVectorMapper::addToken(const std::string& token, const std::vector<float>& vector) {
        if (vector.size() != static_cast<size_t>(vector_dim_)) {
            throw std::invalid_argument("Vector dimension mismatch");
        }
        token_vectors_[token] = vector;
    }

    std::vector<float> BytePairEncodingVectorMapper::getVector(const std::string& token) const {
        auto it = token_vectors_.find(token);
        if (it != token_vectors_.end()) {
            return it->second;
        }
        return std::vector<float>(vector_dim_, 0.0f);
    }

    std::vector<float>& BytePairEncodingVectorMapper::getVectorRef(const std::string& token) {
        auto it = token_vectors_.find(token);
        if (it != token_vectors_.end()) {
            return it->second;
        }
        addToken(token);
        return token_vectors_[token];
    }

    bool BytePairEncodingVectorMapper::hasToken(const std::string& token) const noexcept {
        return token_vectors_.find(token) != token_vectors_.end();
    }

    std::vector<std::string> BytePairEncodingVectorMapper::getTokens() const {
        std::vector<std::string> tokens;
        tokens.reserve(token_vectors_.size());
        for (const auto& pair : token_vectors_) {
            tokens.push_back(pair.first);
        }
        return tokens;
    }

    float BytePairEncodingVectorMapper::getSimilarity(const std::string& token1, const std::string& token2) const {
        auto it1 = token_vectors_.find(token1);
        auto it2 = token_vectors_.find(token2);

        if (it1 == token_vectors_.end() || it2 == token_vectors_.end()) {
            return 0.0f;
        }

        const auto& vec1 = it1->second;
        const auto& vec2 = it2->second;

        float dot = dotProduct(vec1, vec2);
        float mag1 = magnitude(vec1);
        float mag2 = magnitude(vec2);

        if (mag1 == 0.0f || mag2 == 0.0f) {
            return 0.0f;
        }

        return dot / (mag1 * mag2);
    }

    std::vector<std::pair<std::string, float>> BytePairEncodingVectorMapper::findSimilar(const std::string& token, int top_k) const {
        std::vector<std::pair<std::string, float>> similarities;

        if (token_vectors_.find(token) == token_vectors_.end()) {
            return similarities;
        }

        for (const auto& pair : token_vectors_) {
            if (pair.first != token) {
                float sim = getSimilarity(token, pair.first);
                similarities.emplace_back(pair.first, sim);
            }
        }

        std::partial_sort(similarities.begin(),
            similarities.begin() + std::min(top_k, static_cast<int>(similarities.size())),
            similarities.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        if (similarities.size() > static_cast<size_t>(top_k)) {
            similarities.resize(top_k);
        }

        return similarities;
    }

    std::vector<std::vector<float>> BytePairEncodingVectorMapper::encodeToVectors(const std::string& word) const {
        auto tokens = encode(word);
        std::vector<std::vector<float>> vectors;
        vectors.reserve(tokens.size());

        for (const auto& token : tokens) {
            if (hasToken(token)) {
                vectors.push_back(getVector(token));
            }
            else {
                const_cast<BytePairEncodingVectorMapper*>(this)->addToken(token);
                vectors.push_back(getVector(token));
            }
        }

        return vectors;
    }

    bool BytePairEncodingVectorMapper::updateVector(const std::string& token, const std::vector<float>& new_vector) {
        if (new_vector.size() != static_cast<size_t>(vector_dim_)) {
            return false;
        }

        auto it = token_vectors_.find(token);
        if (it != token_vectors_.end()) {
            it->second = new_vector;
            return true;
        }

        return false;
    }

    bool BytePairEncodingVectorMapper::save(const std::string& bpe_filename, const std::string& embeddings_filename) const {
        // Save BPE data
        std::ofstream bpe_out(bpe_filename, std::ios::binary);
        if (!bpe_out) return false;

        uint32_t num_base_tokens = static_cast<uint32_t>(base_tokens_.size());
        bpe_out.write(reinterpret_cast<const char*>(&num_base_tokens), sizeof(num_base_tokens));

        for (const auto& token : base_tokens_) {
            write_string(bpe_out, token);
        }

        uint32_t num_merges = static_cast<uint32_t>(merges_.size());
        bpe_out.write(reinterpret_cast<const char*>(&num_merges), sizeof(num_merges));

        for (const auto& [first, second] : merges_) {
            write_string(bpe_out, first);
            write_string(bpe_out, second);
        }

        if (!bpe_out.good()) return false;

        // Save embeddings
        std::ofstream emb_out(embeddings_filename, std::ios::binary);
        if (!emb_out) return false;

        emb_out.write(reinterpret_cast<const char*>(&vector_dim_), sizeof(vector_dim_));

        size_t vocab_size = token_vectors_.size();
        emb_out.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

        for (const auto& pair : token_vectors_) {
            uint32_t token_len = static_cast<uint32_t>(pair.first.size());
            emb_out.write(reinterpret_cast<const char*>(&token_len), sizeof(token_len));
            emb_out.write(pair.first.c_str(), token_len);
            emb_out.write(reinterpret_cast<const char*>(pair.second.data()),
                vector_dim_ * sizeof(float));
        }

        return emb_out.good();
    }

    bool BytePairEncodingVectorMapper::load(const std::string& bpe_filename, const std::string& embeddings_filename) {
        // Load BPE data
        std::ifstream bpe_in(bpe_filename, std::ios::binary);
        if (!bpe_in) return false;

        uint32_t num_base_tokens = 0;
        bpe_in.read(reinterpret_cast<char*>(&num_base_tokens), sizeof(num_base_tokens));

        base_tokens_.clear();
        for (uint32_t i = 0; i < num_base_tokens; ++i) {
            std::string token = read_string(bpe_in);
            base_tokens_.insert(std::move(token));
        }

        uint32_t num_merges = 0;
        bpe_in.read(reinterpret_cast<char*>(&num_merges), sizeof(num_merges));

        merges_.clear();
        merges_.reserve(num_merges);

        for (uint32_t i = 0; i < num_merges; ++i) {
            std::string first = read_string(bpe_in);
            std::string second = read_string(bpe_in);
            merges_.emplace_back(std::move(first), std::move(second));
        }

        if (!bpe_in.good()) return false;

        // Set training state
        current_merge_step_ = static_cast<int>(merges_.size());
        is_trained_ = current_merge_step_ > 0;

        // Load embeddings
        std::ifstream emb_in(embeddings_filename, std::ios::binary);
        if (!emb_in) return false;

        int loaded_dim;
        emb_in.read(reinterpret_cast<char*>(&loaded_dim), sizeof(loaded_dim));
        if (loaded_dim != vector_dim_) return false;

        size_t vocab_size;
        emb_in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

        token_vectors_.clear();
        token_vectors_.reserve(vocab_size);

        for (size_t i = 0; i < vocab_size; ++i) {
            uint32_t token_len;
            emb_in.read(reinterpret_cast<char*>(&token_len), sizeof(token_len));

            std::string token(token_len, '\0');
            emb_in.read(&token[0], token_len);

            std::vector<float> vector(vector_dim_);
            emb_in.read(reinterpret_cast<char*>(vector.data()), vector_dim_ * sizeof(float));

            token_vectors_[token] = std::move(vector);
        }

        return emb_in.good();
    }

    // Private helper methods

    std::vector<float> BytePairEncodingVectorMapper::initRandomVector() {
        std::vector<float> vec(vector_dim_);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        for (auto& val : vec) {
            val = dist(rng_);
        }

        normalizeVector(vec);
        return vec;
    }

    void BytePairEncodingVectorMapper::normalizeVector(std::vector<float>& vec) {
        float mag = magnitude(vec);
        if (mag > 0.0f) {
            for (auto& val : vec) {
                val /= mag;
            }
        }
    }

    float BytePairEncodingVectorMapper::dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
        float result = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    float BytePairEncodingVectorMapper::magnitude(const std::vector<float>& vec) {
        float sum = 0.0f;
        for (float val : vec) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    void BytePairEncodingVectorMapper::write_string(std::ofstream& out, const std::string& str) {
        uint32_t len = static_cast<uint32_t>(str.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(str.data(), len);
    }

    std::string BytePairEncodingVectorMapper::read_string(std::ifstream& in) {
        uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!in) throw std::runtime_error("Failed to read string length");

        std::string str(len, '\0');
        in.read(&str[0], len);
        if (!in) throw std::runtime_error("Failed to read string data");
        return str;
    }

    void BytePairEncodingVectorMapper::initializeVectorsForNewTokens() {
        // Initialize vectors for all base tokens that don't have vectors yet
        for (const auto& token : base_tokens_) {
            if (token_vectors_.find(token) == token_vectors_.end()) {
                token_vectors_[token] = initRandomVector();
            }
        }
    }

    void BytePairEncodingVectorMapper::updateVectorsAfterMerge(const TokenPair& merged_pair) {
        std::string new_token = merged_pair.first + merged_pair.second;

        // Create vector for new merged token by averaging the component vectors
        if (token_vectors_.find(new_token) == token_vectors_.end()) {
            auto it1 = token_vectors_.find(merged_pair.first);
            auto it2 = token_vectors_.find(merged_pair.second);

            if (it1 != token_vectors_.end() && it2 != token_vectors_.end()) {
                std::vector<float> new_vector(vector_dim_);
                for (int i = 0; i < vector_dim_; ++i) {
                    new_vector[i] = (it1->second[i] + it2->second[i]) * 0.5f;
                }
                normalizeVector(new_vector);
                token_vectors_[new_token] = std::move(new_vector);
            }
            else {
                // Fallback to random vector if components not found
                token_vectors_[new_token] = initRandomVector();
            }
        }
    }

    bool BytePairEncodingVectorMapper::saveCheckpoint(const std::string& checkpoint_prefix, int merge_step) const {
        try {
            std::string bpe_file = getCheckpointFilename(checkpoint_prefix, "bpe");
            std::string emb_file = getCheckpointFilename(checkpoint_prefix, "emb");
            std::string state_file = getCheckpointFilename(checkpoint_prefix, "state");

            // Save BPE and embeddings
            if (!save(bpe_file, emb_file)) {
                return false;
            }

            // Save training state
            std::ofstream state_out(state_file, std::ios::binary);
            if (!state_out) return false;

            state_out.write(reinterpret_cast<const char*>(&merge_step), sizeof(merge_step));
            state_out.write(reinterpret_cast<const char*>(&vector_dim_), sizeof(vector_dim_));
            state_out.write(reinterpret_cast<const char*>(&is_trained_), sizeof(is_trained_));
            state_out.write(reinterpret_cast<const char*>(&min_merge_frequency_), sizeof(min_merge_frequency_));
            state_out.write(reinterpret_cast<const char*>(&save_interval_), sizeof(save_interval_));
            state_out.write(reinterpret_cast<const char*>(&corpus_chunk_size_), sizeof(corpus_chunk_size_));

            return state_out.good();
        }
        catch (const std::exception& e) {
            std::cerr << "Error saving checkpoint: " << e.what() << std::endl;
            return false;
        }
    }

    bool BytePairEncodingVectorMapper::loadCheckpoint(const std::string& checkpoint_prefix) {
        try {
            std::string bpe_file = getCheckpointFilename(checkpoint_prefix, "bpe");
            std::string emb_file = getCheckpointFilename(checkpoint_prefix, "emb");
            std::string state_file = getCheckpointFilename(checkpoint_prefix, "state");

            // Check if checkpoint files exist
            if (!std::filesystem::exists(bpe_file) ||
                !std::filesystem::exists(emb_file) ||
                !std::filesystem::exists(state_file)) {
                return false;
            }

            // Load training state
            std::ifstream state_in(state_file, std::ios::binary);
            if (!state_in) return false;

            int saved_merge_step, saved_vector_dim;
            bool saved_is_trained = false;
            int saved_min_freq = 1, saved_save_interval = 1000;
            size_t saved_chunk_size = 10000;

            state_in.read(reinterpret_cast<char*>(&saved_merge_step), sizeof(saved_merge_step));
            state_in.read(reinterpret_cast<char*>(&saved_vector_dim), sizeof(saved_vector_dim));

            // Read additional fields if available (backward compatibility)
            if (state_in.good()) {
                state_in.read(reinterpret_cast<char*>(&saved_is_trained), sizeof(saved_is_trained));
            }
            if (state_in.good()) {
                state_in.read(reinterpret_cast<char*>(&saved_min_freq), sizeof(saved_min_freq));
            }
            if (state_in.good()) {
                state_in.read(reinterpret_cast<char*>(&saved_save_interval), sizeof(saved_save_interval));
            }
            if (state_in.good()) {
                state_in.read(reinterpret_cast<char*>(&saved_chunk_size), sizeof(saved_chunk_size));
            }

            // Verify compatibility
            if (saved_vector_dim != vector_dim_) {
                std::cerr << "Vector dimension mismatch in checkpoint: "
                    << saved_vector_dim << " vs " << vector_dim_ << std::endl;
                return false;
            }

            // Load BPE and embeddings
            if (!load(bpe_file, emb_file)) {
                return false;
            }

            current_merge_step_ = saved_merge_step;
            is_trained_ = saved_is_trained || (saved_merge_step > 0);
            min_merge_frequency_ = saved_min_freq;
            save_interval_ = saved_save_interval;
            corpus_chunk_size_ = saved_chunk_size;

            std::cout << "Loaded checkpoint with " << merges_.size() << " merges, "
                << token_vectors_.size() << " token vectors" << std::endl;

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading checkpoint: " << e.what() << std::endl;
            return false;
        }
    }

    std::string BytePairEncodingVectorMapper::getCheckpointFilename(const std::string& prefix, const std::string& suffix) const {
        return prefix + "_" + suffix + ".checkpoint";
    }

    void BytePairEncodingVectorMapper::setMinMergeFrequency(int min_freq) {
        min_merge_frequency_ = std::max(1, min_freq);
    }

    void BytePairEncodingVectorMapper::setSaveInterval(int interval) {
        save_interval_ = std::max(1, interval);
    }

    void BytePairEncodingVectorMapper::setCorpusChunkSize(size_t chunk_size) {
        corpus_chunk_size_ = std::max(static_cast<size_t>(1), chunk_size);
    }

    int BytePairEncodingVectorMapper::getMinMergeFrequency() const {
        return min_merge_frequency_;
    }

    int BytePairEncodingVectorMapper::getSaveInterval() const {
        return save_interval_;
    }

    size_t BytePairEncodingVectorMapper::getCorpusChunkSize() const {
        return corpus_chunk_size_;
    }

    int BytePairEncodingVectorMapper::getCurrentMergeStep() const {
        return current_merge_step_;
    }

    size_t BytePairEncodingVectorMapper::getVocabularySize() const {
        return getAllTokens().size();
    }

    size_t BytePairEncodingVectorMapper::getNumMerges() const {
        return merges_.size();
    }

    void BytePairEncodingVectorMapper::printStatistics() const {
        std::cout << "\n=== BPEVM Statistics ===" << std::endl;
        std::cout << "Training status: " << (is_trained_ ? "Trained" : "Not trained") << std::endl;
        std::cout << "Current merge step: " << current_merge_step_ << std::endl;
        std::cout << "Total merges: " << merges_.size() << std::endl;
        std::cout << "Base tokens: " << base_tokens_.size() << std::endl;
        std::cout << "Total vocabulary size: " << getAllTokens().size() << std::endl;
        std::cout << "Token vectors: " << token_vectors_.size() << std::endl;
        std::cout << "Vector dimension: " << vector_dim_ << std::endl;
        std::cout << "Min merge frequency: " << min_merge_frequency_ << std::endl;
        std::cout << "Save interval: " << save_interval_ << std::endl;
        std::cout << "Corpus chunk size: " << corpus_chunk_size_ << std::endl;
        std::cout << "========================" << std::endl;
    }

}  // namespace NNGL