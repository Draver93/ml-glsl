#pragma once

#include <iostream>
#include <string>
#include <chrono>

namespace NNGL {
    enum class LogLevel {
        LL_ERROR = 0,
        LL_WARN = 1,
        LL_INFO = 2,
        LL_DEBUG = 3,
        LL_TRACE = 4
    };

    class Logger {
    public:
        static Logger& getInstance() {
            static Logger instance;
            return instance;
        }

        void log(const std::string& message, LogLevel level = LogLevel::LL_INFO) {
            if (m_enabled && level <= m_currentLevel) {
                std::string prefix = getLevelPrefix(level);
                std::cout << prefix << message << std::endl;
            }
        }

        void setEnabled(bool enabled) {
            m_enabled = enabled;
        }

        bool isEnabled() const {
            return m_enabled;
        }

        void setLogLevel(LogLevel level) {
            m_currentLevel = level;
        }

        LogLevel getLogLevel() const {
            return m_currentLevel;
        }

    private:
        Logger() : m_enabled(true), m_currentLevel(LogLevel::LL_INFO) {}
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        std::string getLevelPrefix(LogLevel level) {
            switch (level) {
                case LogLevel::LL_ERROR: return "[ERROR] ";
                case LogLevel::LL_WARN:  return "[WARN]  ";
                case LogLevel::LL_INFO:  return "[INFO]  ";
                case LogLevel::LL_DEBUG: return "[DEBUG] ";
                case LogLevel::LL_TRACE: return "[TRACE] ";
                default: return "[INFO]  ";
            }
        }

        bool m_enabled;
        LogLevel m_currentLevel;
    };

    class Timer {
        inline static int tabs = 1;
    public:
        Timer(const std::string& name, LogLevel level = LogLevel::LL_TRACE)
            : m_Name(name), m_Level(level), m_Stopped(false) {
            m_Start = std::chrono::high_resolution_clock::now();
            Timer::tabs++;
        }
        ~Timer() {
            Timer::tabs--;
            if (!m_Stopped) stop();
        }
        void stop() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_Start).count();
            std::string tabs(Timer::tabs, '=');
            Logger::getInstance().log(tabs + m_Name + " took " + std::to_string(duration / 1000.0) + " ms", m_Level);
            m_Stopped = true;
        }
    private:
        std::string m_Name;
        LogLevel m_Level;
        std::chrono::high_resolution_clock::time_point m_Start;
        bool m_Stopped;
    };

    // Convenience macros for logging
    #define LOG(message) NNGL::Logger::getInstance().log(message, NNGL::LogLevel::LL_INFO)
    #define LOG_ERROR(message) NNGL::Logger::getInstance().log(message, NNGL::LogLevel::LL_ERROR)
    #define LOG_WARN(message) NNGL::Logger::getInstance().log(message, NNGL::LogLevel::LL_WARN)
    #define LOG_INFO(message) NNGL::Logger::getInstance().log(message, NNGL::LogLevel::LL_INFO)
    #define LOG_DEBUG(message) NNGL::Logger::getInstance().log(message, NNGL::LogLevel::LL_DEBUG)
    #define LOG_TRACE(message) NNGL::Logger::getInstance().log(message, NNGL::LogLevel::LL_TRACE)
} 