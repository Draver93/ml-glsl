#include "BPE.h"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <sstream>

namespace NNGL {

    // PairHash Implementation
    template <typename T1, typename T2>
    std::size_t PairHash::operator()(const std::pair<T1, T2>& p) const noexcept {
        size_t h1 = std::hash<T1>{}(p.first);
        size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }

    // Explicit template instantiation for std::pair<std::string, std::string>
    template std::size_t PairHash::operator() < std::string, std::string > (const std::pair<std::string, std::string>&) const noexcept;

    // VectorStringHash Implementation
    size_t VectorStringHash::operator()(const std::vector<std::string>& vec) const noexcept {
        size_t seed = 0;
        for (const auto& s : vec) {
            seed ^= std::hash<std::string>{}(s)+0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }

    // VectorStringEqual Implementation
    bool VectorStringEqual::operator()(const std::vector<std::string>& lhs, const std::vector<std::string>& rhs) const noexcept {
        return lhs == rhs;
    }

    // BytePairEncoding Implementation

    BytePairEncoding::BytePairEncoding(int max_merges) noexcept
        : max_merges_(max_merges) {}

    void BytePairEncoding::clean_word(std::string& word) noexcept {
        word.erase(std::remove_if(word.begin(), word.end(),
            [](unsigned char c) { return std::ispunct(c); }), word.end());
        std::transform(word.begin(), word.end(), word.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    }

    void BytePairEncoding::build_vocab(const std::vector<std::string>& corpus) {
        vocab_.clear();
        for (auto word : corpus) {
            clean_word(word);
            if (word.empty()) continue;

            std::vector<std::string> tokens;
            tokens.reserve(word.size());
            for (char c : word) {
                tokens.emplace_back(1, c);
            }
            ++vocab_[std::move(tokens)];
        }
    }

    std::unordered_map<BytePairEncoding::TokenPair, int, PairHash> BytePairEncoding::get_pair_frequencies() const noexcept {
        std::unordered_map<TokenPair, int, PairHash> freqs;
        for (const auto& [tokens, freq] : vocab_) {
            if (tokens.size() < 2) continue;
            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                ++freqs[{tokens[i], tokens[i + 1]}];
            }
        }
        return freqs;
    }

    void BytePairEncoding::merge_pair(const TokenPair& pair_to_merge) {
        std::unordered_map<std::vector<std::string>, int, VectorStringHash, VectorStringEqual> new_vocab;

        for (const auto& [tokens, freq] : vocab_) {
            std::vector<std::string> new_tokens;
            new_tokens.reserve(tokens.size());

            for (size_t i = 0; i < tokens.size(); ) {
                if (i + 1 < tokens.size() &&
                    tokens[i] == pair_to_merge.first &&
                    tokens[i + 1] == pair_to_merge.second) {
                    new_tokens.emplace_back(tokens[i] + tokens[i + 1]);
                    i += 2;
                }
                else {
                    new_tokens.push_back(tokens[i]);
                    ++i;
                }
            }
            new_vocab[std::move(new_tokens)] += freq;
        }
        vocab_ = std::move(new_vocab);
    }

    std::vector<std::string> BytePairEncoding::encode(const std::string& word) const noexcept {
        std::vector<std::string> tokens;
        tokens.reserve(word.size());
        for (char c : word) {
            tokens.emplace_back(1, c);
        }

        for (const auto& merge : merges_) {
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
        return tokens;
    }

    void BytePairEncoding::train(const std::vector<std::string>& corpus) {
        build_vocab(corpus);
        merges_.clear();

        for (int i = 0; i < max_merges_; ++i) {
            auto pair_freqs = get_pair_frequencies();
            if (pair_freqs.empty()) break;

            auto best_pair_it = std::max_element(pair_freqs.begin(), pair_freqs.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

            merges_.push_back(best_pair_it->first);
            merge_pair(best_pair_it->first);
        }
    }

    void BytePairEncoding::train(const std::string& filename) {
        std::ifstream input(filename);
        if (!input) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }

        std::vector<std::string> corpus;
        std::string line;
        while (std::getline(input, line)) {
            std::istringstream iss(line);
            std::string word;
            while (iss >> word) {
                clean_word(word);
                if (!word.empty()) corpus.push_back(word);
            }
        }
        train(corpus);
    }

    bool BytePairEncoding::save(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            //throw std::runtime_error("Could not open file for writing: " + filename);
            return false;
        }

        uint32_t num_merges = static_cast<uint32_t>(merges_.size());
        out.write(reinterpret_cast<const char*>(&num_merges), sizeof(num_merges));

        for (const auto& [first, second] : merges_) {
            write_string(out, first);
            write_string(out, second);
        }
        return true;
    }

    bool BytePairEncoding::load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            //throw std::runtime_error("Could not open file for reading: " + filename);
            return false;
        }

        uint32_t num_merges = 0;
        in.read(reinterpret_cast<char*>(&num_merges), sizeof(num_merges));

        merges_.clear();
        merges_.reserve(num_merges);

        for (uint32_t i = 0; i < num_merges; ++i) {
            std::string first = read_string(in);
            std::string second = read_string(in);
            merges_.emplace_back(std::move(first), std::move(second));
        }
        return true;
    }

    void BytePairEncoding::write_string(std::ofstream& out, const std::string& str) {
        uint32_t len = static_cast<uint32_t>(str.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(str.data(), len);
    }

    std::string BytePairEncoding::read_string(std::ifstream& in) {
        uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!in) throw std::runtime_error("Failed to read string length");

        std::string str(len, '\0');
        in.read(&str[0], len);
        if (!in) throw std::runtime_error("Failed to read string data");
        return str;
    }

    std::vector<std::string> BytePairEncoding::split(const std::string& str) {
        std::vector<std::string> tokens;
        std::istringstream iss(str);
        for (std::string token; iss >> token;) {
            tokens.push_back(std::move(token));
        }
        return tokens;
    }

}  // namespace NNGL