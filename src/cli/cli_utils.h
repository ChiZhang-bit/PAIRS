#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rcfs {

class Arguments {
 public:
  Arguments(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      const std::string token(argv[i] == nullptr ? "" : argv[i]);
      if (token.size() <= 2 || token.compare(0, 2, "--") != 0) {
        throw std::invalid_argument("Unexpected positional argument: " + token);
      }

      const size_t equals = token.find('=');
      if (equals != std::string::npos) {
        const std::string key = token.substr(2, equals - 2);
        validate_key(key);
        insert_value(key, token.substr(equals + 1));
        continue;
      }

      const std::string key = token.substr(2);
      validate_key(key);
      if (i + 1 < argc) {
        const std::string next(argv[i + 1] == nullptr ? "" : argv[i + 1]);
        if (next.compare(0, 2, "--") != 0) {
          insert_value(key, next);
          ++i;
          continue;
        }
      }
      insert_flag(key);
    }
  }

  bool has(const std::string& key) const {
    return values_.find(key) != values_.end() ||
           flags_.find(key) != flags_.end();
  }

  bool flag(const std::string& key) const { return get_bool(key, false); }

  bool get_bool(const std::string& key, bool fallback) const {
    if (flags_.find(key) != flags_.end()) return true;
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    const std::string value = lowercase(trim(it->second));
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
      return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
      return false;
    }
    throw std::invalid_argument("Invalid boolean value for --" + key +
                                ": " + it->second);
  }

  std::string require(const std::string& key) const {
    const auto it = values_.find(key);
    if (it == values_.end()) {
      if (flags_.find(key) != flags_.end()) {
        throw std::invalid_argument("Option --" + key + " requires a value");
      }
      throw std::invalid_argument("Missing required option --" + key);
    }
    return it->second;
  }

  std::string get(const std::string& key,
                  const std::string& fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
  }

  uint32_t get_u32(const std::string& key, uint32_t fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback
                               : parse_u32(it->second, "--" + key);
  }

  uint32_t require_u32(const std::string& key) const {
    return parse_u32(require(key), "--" + key);
  }

  double get_double(const std::string& key, double fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback
                               : parse_double(it->second, "--" + key);
  }

  double require_double(const std::string& key) const {
    return parse_double(require(key), "--" + key);
  }

  float get_float(const std::string& key, float fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback
                               : parse_float(it->second, "--" + key);
  }

  float require_float(const std::string& key) const {
    return parse_float(require(key), "--" + key);
  }

  std::vector<std::string> get_csv(
      const std::string& key,
      std::vector<std::string> fallback = {}) const {
    const auto it = values_.find(key);
    return it == values_.end() ? std::move(fallback)
                               : split_csv(it->second, "--" + key);
  }

  std::vector<uint32_t> get_u32_list(
      const std::string& key, std::vector<uint32_t> fallback = {}) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    const std::vector<std::string> items = split_csv(it->second, "--" + key);
    std::vector<uint32_t> values;
    values.reserve(items.size());
    for (const std::string& item : items) {
      values.push_back(parse_u32(item, "--" + key));
    }
    return values;
  }

  std::vector<double> get_double_list(
      const std::string& key, std::vector<double> fallback = {}) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    const std::vector<std::string> items = split_csv(it->second, "--" + key);
    std::vector<double> values;
    values.reserve(items.size());
    for (const std::string& item : items) {
      values.push_back(parse_double(item, "--" + key));
    }
    return values;
  }

  std::vector<float> get_float_list(
      const std::string& key, std::vector<float> fallback = {}) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    const std::vector<std::string> items = split_csv(it->second, "--" + key);
    std::vector<float> values;
    values.reserve(items.size());
    for (const std::string& item : items) {
      values.push_back(parse_float(item, "--" + key));
    }
    return values;
  }

 private:
  static void validate_key(const std::string& key) {
    if (key.empty()) throw std::invalid_argument("Empty command-line option");
    for (char character : key) {
      const bool valid =
          (character >= 'a' && character <= 'z') ||
          (character >= 'A' && character <= 'Z') ||
          (character >= '0' && character <= '9') || character == '-' ||
          character == '_';
      if (!valid) {
        throw std::invalid_argument("Invalid command-line option --" + key);
      }
    }
  }

  void ensure_new(const std::string& key) const {
    if (has(key)) {
      throw std::invalid_argument("Duplicate command-line option --" + key);
    }
  }

  void insert_value(const std::string& key, const std::string& value) {
    ensure_new(key);
    values_.emplace(key, value);
  }

  void insert_flag(const std::string& key) {
    ensure_new(key);
    flags_.insert(key);
  }

  static std::string trim(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() &&
           (value[begin] == ' ' || value[begin] == '\t' ||
            value[begin] == '\r' || value[begin] == '\n')) {
      ++begin;
    }
    size_t end = value.size();
    while (end > begin &&
           (value[end - 1] == ' ' || value[end - 1] == '\t' ||
            value[end - 1] == '\r' || value[end - 1] == '\n')) {
      --end;
    }
    return value.substr(begin, end - begin);
  }

  static std::string lowercase(std::string value) {
    for (char& character : value) {
      if (character >= 'A' && character <= 'Z') {
        character = static_cast<char>(character - 'A' + 'a');
      }
    }
    return value;
  }

  static uint32_t parse_u32(const std::string& raw,
                            const std::string& label) {
    const std::string value = trim(raw);
    if (value.empty() || value.front() == '-' || value.front() == '+') {
      throw std::invalid_argument("Invalid uint32 value for " + label +
                                  ": " + raw);
    }
    uint32_t parsed = 0;
    const char* begin = value.data();
    const char* end = value.data() + value.size();
    const auto result = std::from_chars(begin, end, parsed, 10);
    if (result.ec != std::errc() || result.ptr != end) {
      throw std::invalid_argument("Invalid uint32 value for " + label +
                                  ": " + raw);
    }
    return parsed;
  }

  static double parse_double(const std::string& raw,
                             const std::string& label) {
    const std::string value = trim(raw);
    if (value.empty()) {
      throw std::invalid_argument("Invalid floating-point value for " + label);
    }
    size_t consumed = 0;
    double parsed = 0.0;
    try {
      parsed = std::stod(value, &consumed);
    } catch (const std::exception&) {
      throw std::invalid_argument("Invalid floating-point value for " + label +
                                  ": " + raw);
    }
    if (consumed != value.size() || !std::isfinite(parsed)) {
      throw std::invalid_argument("Invalid floating-point value for " + label +
                                  ": " + raw);
    }
    return parsed;
  }

  static float parse_float(const std::string& raw,
                           const std::string& label) {
    const double parsed = parse_double(raw, label);
    if (parsed < -static_cast<double>(std::numeric_limits<float>::max()) ||
        parsed > static_cast<double>(std::numeric_limits<float>::max())) {
      throw std::invalid_argument("Float value is out of range for " + label +
                                  ": " + raw);
    }
    return static_cast<float>(parsed);
  }

  static std::vector<std::string> split_csv(const std::string& raw,
                                            const std::string& label) {
    if (trim(raw).empty()) return {};
    std::vector<std::string> items;
    size_t begin = 0;
    while (begin <= raw.size()) {
      const size_t comma = raw.find(',', begin);
      const size_t end = comma == std::string::npos ? raw.size() : comma;
      const std::string item = trim(raw.substr(begin, end - begin));
      if (item.empty()) {
        throw std::invalid_argument("Empty item in comma-separated value for " +
                                    label);
      }
      items.push_back(item);
      if (comma == std::string::npos) break;
      begin = comma + 1;
    }
    return items;
  }

  std::unordered_map<std::string, std::string> values_;
  std::unordered_set<std::string> flags_;
};

inline void ensure_parent_directory(const std::string& path) {
  const std::filesystem::path parent =
      std::filesystem::path(path).parent_path();
  if (parent.empty()) return;
  std::error_code error;
  std::filesystem::create_directories(parent, error);
  if (error) {
    throw std::runtime_error("Cannot create parent directory '" +
                             parent.string() + "': " + error.message());
  }
}

inline void ensure_prefix_parent(const std::string& prefix) {
  ensure_parent_directory(prefix);
}

}  // namespace rcfs
