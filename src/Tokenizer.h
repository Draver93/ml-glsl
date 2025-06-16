#pragma once
#include <iostream>
#include <unordered_map>

namespace NNGL {
    struct TrieNode {
        TrieNode() = default;
        ~TrieNode();

        std::unordered_map<char, TrieNode*> children;
        bool isEndOfWord = false;
        int tokenId = -1;
        int frequency = 0;
    };

    class Trie {
    public:
        Trie();
        ~Trie();

        void insert(const std::string& word);
        int get_token_id(const std::string& token) const;
        std::string get_token_by_id(int id) const;

        bool search(const std::string& word) const;
        bool startsWith(const std::string& prefix) const;

        void prune();
        void prune_stddev_threshold(double n_stddev);

        void bulk_insert_from_file(const std::string& filename);

        std::vector<std::string> tokenize(const std::string& line) const;

        void clear();

        void save_tokens(const std::string& filename) const;
        void load_tokens(const std::string& filename);

    private:
        TrieNode* root;
        int nextTokenId;

        bool find_token_by_id(TrieNode* node, int id, std::string& current, std::string& result) const;
        bool pruneHelper(TrieNode* node);
        void collectFrequencies(std::vector<std::pair<std::string, int>>& out) const;
        void collectFrequenciesHelper(TrieNode* node, const std::string& path, std::vector<std::pair<std::string, int>>& out) const;
        bool set_frequency(const std::string& word, int freq);
    };
}

