#define NOMINMAX

#include "Tokenizer.h"

#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <iostream>
#include <algorithm>

/*namespace NNGL {
    void TokenVectorMapper::initializeFromBPE(const BytePairEncoding& bpe) {
        auto tokens = bpe.getAllTokens();
        initializeFromTokens(tokens);
    }

    void TokenVectorMapper::initializeFromTokens(const std::set<std::string>& tokens) {
        token_vectors_.clear();
        token_vectors_.reserve(tokens.size());

        for (const auto& token : tokens) {
            token_vectors_[token] = initRandomVector();
        }
    }

    void TokenVectorMapper::addToken(const std::string& token) {
        if (token_vectors_.find(token) == token_vectors_.end()) {
            token_vectors_[token] = initRandomVector();
        }
    }

    void TokenVectorMapper::addToken(const std::string& token, const std::vector<float>& vector) {
        if (vector.size() != static_cast<size_t>(vector_dim_)) {
            throw std::invalid_argument("Vector dimension mismatch");
        }
        token_vectors_[token] = vector;
    }

    std::vector<float> TokenVectorMapper::getVector(const std::string& token) const {
        auto it = token_vectors_.find(token);
        if (it != token_vectors_.end()) {
            return it->second;
        }

        // Return zero vector for unknown tokens
        return std::vector<float>(vector_dim_, 0.0f);
    }

    std::vector<float>& TokenVectorMapper::getVectorRef(const std::string& token) {
        auto it = token_vectors_.find(token);
        if (it != token_vectors_.end()) {
            return it->second;
        }

        // Add token if it doesn't exist
        addToken(token);
        return token_vectors_[token];
    }

    bool TokenVectorMapper::hasToken(const std::string& token) const noexcept {
        return token_vectors_.find(token) != token_vectors_.end();
    }

    std::vector<std::string> TokenVectorMapper::getTokens() const {
        std::vector<std::string> tokens;
        tokens.reserve(token_vectors_.size());

        for (const auto& pair : token_vectors_) {
            tokens.push_back(pair.first);
        }

        return tokens;
    }

    bool TokenVectorMapper::saveEmbeddings(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            return false;
        }

        // Write header
        out.write(reinterpret_cast<const char*>(&vector_dim_), sizeof(vector_dim_));

        size_t vocab_size = token_vectors_.size();
        out.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

        // Write token-vector pairs
        for (const auto& pair : token_vectors_) {
            // Write token length and token
            uint32_t token_len = static_cast<uint32_t>(pair.first.size());
            out.write(reinterpret_cast<const char*>(&token_len), sizeof(token_len));
            out.write(pair.first.c_str(), token_len);

            // Write vector
            out.write(reinterpret_cast<const char*>(pair.second.data()),
                vector_dim_ * sizeof(float));
        }

        return out.good();
    }

    bool TokenVectorMapper::loadEmbeddings(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            return false;
        }

        // Read header
        int loaded_dim;
        in.read(reinterpret_cast<char*>(&loaded_dim), sizeof(loaded_dim));

        if (loaded_dim != vector_dim_) {
            return false; // Dimension mismatch
        }

        size_t vocab_size;
        in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

        token_vectors_.clear();
        token_vectors_.reserve(vocab_size);

        // Read token-vector pairs
        for (size_t i = 0; i < vocab_size; ++i) {
            // Read token
            uint32_t token_len;
            in.read(reinterpret_cast<char*>(&token_len), sizeof(token_len));

            std::string token(token_len, '\0');
            in.read(&token[0], token_len);

            // Read vector
            std::vector<float> vector(vector_dim_);
            in.read(reinterpret_cast<char*>(vector.data()), vector_dim_ * sizeof(float));

            token_vectors_[token] = std::move(vector);
        }

        return in.good();
    }

    bool TokenVectorMapper::saveWithBPE(const BytePairEncoding& bpe, const std::string& bpe_filename,
        const std::string& embeddings_filename) const {
        return bpe.save(bpe_filename) && saveEmbeddings(embeddings_filename);
    }

    bool TokenVectorMapper::loadWithBPE(BytePairEncoding& bpe, const std::string& bpe_filename,
        const std::string& embeddings_filename) {
        if (!bpe.load(bpe_filename)) {
            return false;
        }

        if (!loadEmbeddings(embeddings_filename)) {
            return false;
        }

        // Ensure all BPE tokens have embeddings
        auto bpe_tokens = bpe.getAllTokens();
        for (const auto& token : bpe_tokens) {
            if (!hasToken(token)) {
                addToken(token);
            }
        }

        return true;
    }

    bool TokenVectorMapper::updateVector(const std::string& token, const std::vector<float>& new_vector) {
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

    float TokenVectorMapper::getSimilarity(const std::string& token1, const std::string& token2) const {
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

    std::vector<std::pair<std::string, float>> TokenVectorMapper::findSimilar(const std::string& token, int top_k) const {
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

        // Sort by similarity (descending)
        std::partial_sort(similarities.begin(),
            similarities.begin() + std::min(top_k, static_cast<int>(similarities.size())),
            similarities.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        if (similarities.size() > static_cast<size_t>(top_k)) {
            similarities.resize(top_k);
        }

        return similarities;
    }

    std::vector<std::vector<float>> TokenVectorMapper::encodeToVectors(const BytePairEncoding& bpe, const std::string& word) const {
        auto tokens = bpe.encode(word);
        std::vector<std::vector<float>> vectors;
        vectors.reserve(tokens.size());

        for (const auto& token : tokens) {
            if (hasToken(token)) {
                vectors.push_back(getVector(token));
            }
            else {
                // For unknown tokens, add them with random vectors
                const_cast<TokenVectorMapper*>(this)->addToken(token);
                vectors.push_back(getVector(token));
            }
        }

        return vectors;
    }

    std::vector<float> TokenVectorMapper::initRandomVector() const {
        std::vector<float> vec(vector_dim_);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        for (auto& val : vec) {
            val = dist(rng_);
        }

        normalizeVector(vec);
        return vec;
    }

    void TokenVectorMapper::normalizeVector(std::vector<float>& vec) {
        float mag = magnitude(vec);
        if (mag > 0.0f) {
            for (auto& val : vec) {
                val /= mag;
            }
        }
    }

    float TokenVectorMapper::dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
        float result = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    float TokenVectorMapper::magnitude(const std::vector<float>& vec) {
        float sum = 0.0f;
        for (float val : vec) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }
}*/

