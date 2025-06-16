#include "Tokenizer.h"

#include <fstream>
#include <string>
#include <sstream>


namespace NNGL {
    TrieNode::~TrieNode() {
        for (auto& [_, child] : children) {
            delete child;
        }
    }

    Trie::Trie() : root(new TrieNode()), nextTokenId(0) {}

    Trie::~Trie() {
        delete root;
    }

    void Trie::insert(const std::string& word) {
        TrieNode* node = root;
        for (char c : word) {
            if (!node->children.count(c)) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        if (!node->isEndOfWord) {
            node->isEndOfWord = true;
            node->tokenId = nextTokenId++;
            node->frequency = 0;
        }
        node->frequency++;
    }

    int Trie::get_token_id(const std::string& token) const {
        TrieNode* node = root;
        for (char c : token) {
            if (!node->children.count(c))
                return -1;
            node = node->children.at(c);
        }
        return node->isEndOfWord ? node->tokenId : -1;
    }

    std::string Trie::get_token_by_id(int id) const {
        std::string result, current;
        if (find_token_by_id(root, id, current, result)) {
            return result;
        }
        return "";
    }

    bool Trie::find_token_by_id(TrieNode* node, int id, std::string& current, std::string& result) const {
        if (node->isEndOfWord && node->tokenId == id) {
            result = current;
            return true;
        }
        for (auto& [ch, child] : node->children) {
            current.push_back(ch);
            if (find_token_by_id(child, id, current, result)) {
                return true;
            }
            current.pop_back();
        }
        return false;
    }

    bool Trie::search(const std::string& word) const {
        TrieNode* node = root;
        for (char ch : word) {
            if (!node->children.count(ch)) {
                return false;
            }
            node = node->children.at(ch);
        }
        return node->isEndOfWord;
    }

    bool Trie::startsWith(const std::string& prefix) const {
        TrieNode* node = root;
        for (char ch : prefix) {
            if (!node->children.count(ch)) {
                return false;
            }
            node = node->children.at(ch);
        }
        return true;
    }

    void Trie::prune() {
        pruneHelper(root);
    }

    bool Trie::pruneHelper(TrieNode* node) {
        std::vector<char> toErase;

        for (auto& [ch, child] : node->children) {
            bool keep = pruneHelper(child);
            if (!keep) {
                delete child;
                toErase.push_back(ch);
            }
        }
        for (char ch : toErase) {
            node->children.erase(ch);
        }
        return node->isEndOfWord || !node->children.empty();
    }

    void Trie::collectFrequencies(std::vector<std::pair<std::string, int>>& out) const {
        collectFrequenciesHelper(root, "", out);
    }

    void Trie::collectFrequenciesHelper(TrieNode* node, const std::string& path, std::vector<std::pair<std::string, int>>& out) const {
        if (node->isEndOfWord) {
            out.emplace_back(path, node->frequency);
        }
        for (auto& [ch, child] : node->children) {
            collectFrequenciesHelper(child, path + ch, out);
        }
    }

    bool Trie::set_frequency(const std::string& word, int freq) {
        TrieNode* node = root;
        for (char c : word) {
            if (!node->children.count(c)) return false;
            node = node->children[c];
        }
        if (node->isEndOfWord) {
            node->frequency = freq;
            return true;
        }
        return false;
    }

    void Trie::prune_stddev_threshold(double n_stddev) {
        std::vector<std::pair<std::string, int>> tokens;
        collectFrequencies(tokens);

        if (tokens.empty()) return;

        double sum = 0;
        for (const auto& [_, freq] : tokens)
            sum += freq;
        double mean = sum / tokens.size();

        double variance = 0;
        for (const auto& [_, freq] : tokens)
            variance += (freq - mean) * (freq - mean);
        double stddev = sqrt(variance / tokens.size());

        double threshold = mean - n_stddev * stddev;

        std::cout << "[Prune Info] Mean = " << mean << ", StdDev = " << stddev
            << ", Threshold = " << threshold << std::endl;

        for (const auto& [word, freq] : tokens) {
            if (freq < threshold) {
                set_frequency(word, 0);
            }
        }

        prune();
    }

    void Trie::bulk_insert_from_file(const std::string& filename) {
        std::ifstream infile(filename);
        std::string line;
        size_t counter = 0;

        while (std::getline(infile, line)) {
            std::vector<std::string> words = tokenize(line);
            for (const std::string& word : words) {
                insert(word);
            }
            if (++counter % 10000 == 0) {
                std::cout << "Processed " << counter << " lines...\n";
            }
        }
    }

    std::vector<std::string> Trie::tokenize(const std::string& line) const {
        std::vector<std::string> tokens;
        std::string token;
        for (char c : line) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                token += std::tolower(static_cast<unsigned char>(c));
            }
            else if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        }
        if (!token.empty()) tokens.push_back(token);
        return tokens;
    }

    void Trie::clear() {
        delete root;
        root = new TrieNode();
        nextTokenId = 0;
    }
    // Save all tokens with their IDs and frequencies into a file
    void Trie::save_tokens(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out.is_open()) {
            std::cerr << "Failed to open file for saving tokens\n";
            return;
        }

        std::vector<std::pair<std::string, int>> tokens;
        collectFrequencies(tokens);

        for (const auto& [token, freq] : tokens) {
            int id = get_token_id(token);
            if (id != -1) {
                out << id << " " << token << " " << freq << "\n";
            }
        }
        out.close();
    }

    // Load tokens from file and rebuild the trie with correct ids and frequencies
    void Trie::load_tokens(const std::string& filename) {
        std::ifstream in(filename);
        if (!in.is_open()) {
            std::cerr << "Failed to open file for loading tokens\n";
            return;
        }

        clear();

        std::string line;
        int maxId = -1;
        while (std::getline(in, line)) {
            std::istringstream iss(line);
            int id; std::string token; int freq;
            if (!(iss >> id >> token >> freq)) continue;

            TrieNode* node = root;
            for (char c : token) {
                if (!node->children.count(c)) {
                    node->children[c] = new TrieNode();
                }
                node = node->children[c];
            }
            node->isEndOfWord = true;
            node->tokenId = id;
            node->frequency = freq;

            if (id > maxId) maxId = id;
        }

        nextTokenId = maxId + 1;
        in.close();
    }

}

