#include "rcfs/predicate.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace rcfs {
namespace {

enum class JsonType : uint8_t {
  kNull,
  kBoolean,
  kNumber,
  kString,
  kArray,
  kObject,
};

struct JsonValue {
  JsonType type = JsonType::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<JsonValue> array;
  std::vector<std::pair<std::string, JsonValue>> object;
};

class JsonParser {
 public:
  JsonParser(const std::string& text, uint32_t max_depth)
      : text_(text), max_depth_(max_depth) {}

  JsonValue parse_document() {
    JsonValue value = parse_value(0);
    skip_space();
    if (cursor_ != text_.size()) fail("Unexpected bytes after JSON value");
    return value;
  }

 private:
  [[noreturn]] void fail(const std::string& message) const {
    throw std::runtime_error(message + " at byte " +
                             std::to_string(cursor_));
  }

  void skip_space() {
    while (cursor_ < text_.size()) {
      const char ch = text_[cursor_];
      if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') break;
      ++cursor_;
    }
  }

  bool consume(char expected) {
    skip_space();
    if (cursor_ < text_.size() && text_[cursor_] == expected) {
      ++cursor_;
      return true;
    }
    return false;
  }

  void expect(char expected) {
    if (!consume(expected)) {
      fail(std::string("Expected '") + expected + "'");
    }
  }

  bool consume_literal(const char* literal) {
    size_t length = 0;
    while (literal[length] != '\0') ++length;
    if (text_.compare(cursor_, length, literal) != 0) return false;
    cursor_ += length;
    return true;
  }

  static uint32_t hex_digit(char ch) {
    if (ch >= '0' && ch <= '9') return static_cast<uint32_t>(ch - '0');
    if (ch >= 'a' && ch <= 'f') return 10U + static_cast<uint32_t>(ch - 'a');
    if (ch >= 'A' && ch <= 'F') return 10U + static_cast<uint32_t>(ch - 'A');
    return std::numeric_limits<uint32_t>::max();
  }

  uint32_t parse_hex_quad() {
    if (text_.size() - cursor_ < 4) fail("Truncated Unicode escape");
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const uint32_t digit = hex_digit(text_[cursor_++]);
      if (digit == std::numeric_limits<uint32_t>::max()) {
        fail("Invalid Unicode escape");
      }
      value = (value << 4U) | digit;
    }
    return value;
  }

  static void append_utf8(uint32_t code_point, std::string* output) {
    if (code_point <= 0x7FU) {
      output->push_back(static_cast<char>(code_point));
    } else if (code_point <= 0x7FFU) {
      output->push_back(static_cast<char>(0xC0U | (code_point >> 6U)));
      output->push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
    } else if (code_point <= 0xFFFFU) {
      output->push_back(static_cast<char>(0xE0U | (code_point >> 12U)));
      output->push_back(
          static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
      output->push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
    } else {
      output->push_back(static_cast<char>(0xF0U | (code_point >> 18U)));
      output->push_back(
          static_cast<char>(0x80U | ((code_point >> 12U) & 0x3FU)));
      output->push_back(
          static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
      output->push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
    }
  }

  std::string parse_string() {
    skip_space();
    if (cursor_ >= text_.size() || text_[cursor_] != '"') {
      fail("Expected JSON string");
    }
    ++cursor_;
    std::string result;
    while (cursor_ < text_.size()) {
      const unsigned char ch = static_cast<unsigned char>(text_[cursor_++]);
      if (ch == '"') return result;
      if (ch < 0x20U) fail("Unescaped control byte in JSON string");
      if (ch != '\\') {
        result.push_back(static_cast<char>(ch));
        continue;
      }
      if (cursor_ >= text_.size()) fail("Truncated JSON string escape");
      const char escaped = text_[cursor_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          result.push_back(escaped);
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'u': {
          uint32_t code_point = parse_hex_quad();
          if (code_point >= 0xD800U && code_point <= 0xDBFFU) {
            if (text_.size() - cursor_ < 6 || text_[cursor_] != '\\' ||
                text_[cursor_ + 1] != 'u') {
              fail("High surrogate lacks a low surrogate");
            }
            cursor_ += 2;
            const uint32_t low = parse_hex_quad();
            if (low < 0xDC00U || low > 0xDFFFU) {
              fail("Invalid low surrogate");
            }
            code_point = 0x10000U + ((code_point - 0xD800U) << 10U) +
                         (low - 0xDC00U);
          } else if (code_point >= 0xDC00U && code_point <= 0xDFFFU) {
            fail("Unexpected low surrogate");
          }
          append_utf8(code_point, &result);
          break;
        }
        default:
          fail("Invalid JSON string escape");
      }
    }
    fail("Unterminated JSON string");
  }

  JsonValue parse_number() {
    skip_space();
    const size_t begin = cursor_;
    if (cursor_ < text_.size() && text_[cursor_] == '-') ++cursor_;
    if (cursor_ >= text_.size()) fail("Truncated JSON number");
    if (text_[cursor_] == '0') {
      ++cursor_;
      if (cursor_ < text_.size() && text_[cursor_] >= '0' &&
          text_[cursor_] <= '9') {
        fail("Leading zero in JSON number");
      }
    } else {
      if (text_[cursor_] < '1' || text_[cursor_] > '9') {
        fail("Invalid JSON number");
      }
      while (cursor_ < text_.size() && text_[cursor_] >= '0' &&
             text_[cursor_] <= '9') {
        ++cursor_;
      }
    }
    if (cursor_ < text_.size() && text_[cursor_] == '.') {
      ++cursor_;
      const size_t fraction_begin = cursor_;
      while (cursor_ < text_.size() && text_[cursor_] >= '0' &&
             text_[cursor_] <= '9') {
        ++cursor_;
      }
      if (cursor_ == fraction_begin) fail("Empty JSON number fraction");
    }
    if (cursor_ < text_.size() &&
        (text_[cursor_] == 'e' || text_[cursor_] == 'E')) {
      ++cursor_;
      if (cursor_ < text_.size() &&
          (text_[cursor_] == '+' || text_[cursor_] == '-')) {
        ++cursor_;
      }
      const size_t exponent_begin = cursor_;
      while (cursor_ < text_.size() && text_[cursor_] >= '0' &&
             text_[cursor_] <= '9') {
        ++cursor_;
      }
      if (cursor_ == exponent_begin) fail("Empty JSON number exponent");
    }
    const std::string encoded = text_.substr(begin, cursor_ - begin);
    errno = 0;
    char* end = nullptr;
    const double number = std::strtod(encoded.c_str(), &end);
    if (errno == ERANGE || end != encoded.c_str() + encoded.size() ||
        !std::isfinite(number)) {
      fail("Non-finite or invalid JSON number");
    }
    JsonValue value;
    value.type = JsonType::kNumber;
    value.number = number;
    return value;
  }

  JsonValue parse_array(uint32_t depth) {
    JsonValue value;
    value.type = JsonType::kArray;
    expect('[');
    if (consume(']')) return value;
    while (true) {
      value.array.push_back(parse_value(depth + 1));
      if (consume(']')) return value;
      expect(',');
    }
  }

  JsonValue parse_object(uint32_t depth) {
    JsonValue value;
    value.type = JsonType::kObject;
    expect('{');
    if (consume('}')) return value;
    std::unordered_set<std::string> keys;
    while (true) {
      const std::string key = parse_string();
      if (!keys.insert(key).second) fail("Duplicate JSON object key");
      expect(':');
      value.object.emplace_back(key, parse_value(depth + 1));
      if (consume('}')) return value;
      expect(',');
    }
  }

  JsonValue parse_value(uint32_t depth) {
    if (depth > max_depth_) fail("JSON nesting exceeds configured maximum");
    skip_space();
    if (cursor_ >= text_.size()) fail("Expected JSON value");
    const char ch = text_[cursor_];
    if (ch == '{') return parse_object(depth);
    if (ch == '[') return parse_array(depth);
    if (ch == '"') {
      JsonValue value;
      value.type = JsonType::kString;
      value.string = parse_string();
      return value;
    }
    if (ch == '-' || (ch >= '0' && ch <= '9')) return parse_number();
    JsonValue value;
    if (consume_literal("true")) {
      value.type = JsonType::kBoolean;
      value.boolean = true;
      return value;
    }
    if (consume_literal("false")) {
      value.type = JsonType::kBoolean;
      value.boolean = false;
      return value;
    }
    if (consume_literal("null")) {
      value.type = JsonType::kNull;
      return value;
    }
    fail("Unsupported JSON token");
  }

  const std::string& text_;
  uint32_t max_depth_;
  size_t cursor_ = 0;
};

const JsonValue* find_member(const JsonValue& object, const std::string& name) {
  if (object.type != JsonType::kObject) return nullptr;
  for (const auto& item : object.object) {
    if (item.first == name) return &item.second;
  }
  return nullptr;
}

uint32_t json_uint32(const JsonValue& value, const std::string& field) {
  if (value.type != JsonType::kNumber || value.number < 0.0 ||
      value.number > std::numeric_limits<uint32_t>::max() ||
      std::floor(value.number) != value.number) {
    throw std::runtime_error(field + " must be a uint32 integer");
  }
  return static_cast<uint32_t>(value.number);
}

PredicateNode parse_predicate_node(const JsonValue& value,
                                   const Schema& schema, uint32_t depth,
                                   uint32_t max_depth) {
  if (depth > max_depth) {
    throw std::runtime_error("Predicate AST exceeds configured depth");
  }
  if (value.type != JsonType::kObject || value.object.size() != 1) {
    throw std::runtime_error(
        "Every predicate node must be an object with exactly one key");
  }
  const std::string& key = value.object.front().first;
  const JsonValue& operand = value.object.front().second;
  if (key == "NOT") {
    throw std::runtime_error("NOT predicates are not supported");
  }
  if (key == "AND" || key == "OR") {
    if (operand.type != JsonType::kArray || operand.array.size() < 2) {
      throw std::runtime_error(key + " must contain at least two children");
    }
    PredicateNode node;
    node.kind = key == "AND" ? PredicateNodeKind::kAnd
                             : PredicateNodeKind::kOr;
    node.children.reserve(operand.array.size());
    for (const JsonValue& child : operand.array) {
      node.children.push_back(
          parse_predicate_node(child, schema, depth + 1, max_depth));
    }
    return node;
  }

  const auto numeric_it =
      std::find(schema.numeric_names.begin(), schema.numeric_names.end(), key);
  if (numeric_it != schema.numeric_names.end()) {
    if (operand.type != JsonType::kArray || operand.array.size() != 2 ||
        operand.array[0].type != JsonType::kNumber ||
        operand.array[1].type != JsonType::kNumber) {
      throw std::runtime_error("Numeric predicate '" + key +
                               "' must be a two-number [lower,upper] array");
    }
    const double lower = operand.array[0].number;
    const double upper = operand.array[1].number;
    if (lower > upper) {
      throw std::runtime_error("Numeric predicate lower bound exceeds upper "
                               "bound for attribute " +
                               key);
    }
    PredicateNode node;
    node.kind = PredicateNodeKind::kNumericRange;
    node.numeric_range.attribute = static_cast<NumericAttrId>(
        numeric_it - schema.numeric_names.begin());
    node.numeric_range.lower = lower;
    node.numeric_range.upper = upper;
    return node;
  }

  const auto categorical_it = std::find(schema.categorical_names.begin(),
                                        schema.categorical_names.end(), key);
  if (categorical_it != schema.categorical_names.end()) {
    std::string value_name;
    if (operand.type == JsonType::kString) {
      value_name = operand.string;
    } else if (operand.type == JsonType::kBoolean) {
      value_name = operand.boolean ? "true" : "false";
      const auto attribute = static_cast<CategoricalAttrId>(categorical_it - schema.categorical_names.begin());
      const auto& names = schema.categorical_value_names[attribute];
      // Accept the title-case labels used by the legacy bool field.
      if (key == "bool" && std::find(names.begin(), names.end(), value_name) == names.end())
        value_name = operand.boolean ? "True" : "False";
    } else {
      throw std::runtime_error("Categorical predicate '" + key +
                               "' must use a string or boolean value");
    }
    const CategoricalAttrId attribute = static_cast<CategoricalAttrId>(
        categorical_it - schema.categorical_names.begin());
    PredicateNode node;
    node.kind = PredicateNodeKind::kCategoricalEquality;
    node.categorical_equality.attribute = attribute;
    node.categorical_equality.value = schema.value_id(attribute, value_name);
    return node;
  }
  throw std::runtime_error("Unknown predicate attribute/operator: " + key);
}

PredicateRecord parse_record(const JsonValue& root, const Schema& schema,
                             uint32_t physical_row, uint32_t max_ast_depth) {
  if (root.type != JsonType::kObject) {
    throw std::runtime_error("JSONL record must be an object");
  }
  // Query IDs identify vector rows. source_query_id below is provenance only.
  const JsonValue* query_id_value = find_member(root, "query_id");
  const JsonValue* local_id_value = find_member(root, "local_query_id");
  if (query_id_value == nullptr && local_id_value == nullptr) {
    throw std::runtime_error(
        "Predicate record needs query_id or local_query_id");
  }
  if (query_id_value != nullptr &&
      json_uint32(*query_id_value, "query_id") != physical_row) {
    throw std::runtime_error("query_id does not match query-vector row " +
                             std::to_string(physical_row));
  }
  if (local_id_value != nullptr &&
      json_uint32(*local_id_value, "local_query_id") != physical_row) {
    throw std::runtime_error("local_query_id does not match query-vector row " +
                             std::to_string(physical_row));
  }

  const JsonValue* predicate_value = find_member(root, "predicate");
  if (predicate_value == nullptr) {
    throw std::runtime_error("Predicate record lacks a predicate field");
  }
  PredicateRecord record;
  record.physical_query_id = physical_row;
  const JsonValue* source_id_value = find_member(root, "source_query_id");
  if (source_id_value != nullptr) {
    record.source_query_id =
        json_uint32(*source_id_value, "source_query_id");
    record.has_source_query_id = true;
  } else {
    record.source_query_id = physical_row;
  }
  const JsonValue* template_value = find_member(root, "template");
  if (template_value != nullptr) {
    if (template_value->type != JsonType::kString) {
      throw std::runtime_error("template must be a string when present");
    }
    record.template_name = template_value->string;
  }
  record.predicate = parse_predicate_node(*predicate_value, schema, 1,
                                          max_ast_depth);
  return record;
}

}  // namespace

std::vector<PredicateRecord> read_predicate_jsonl(
    const std::string& path, const Schema& schema,
    size_t expected_query_count, uint32_t max_ast_depth) {
  schema.validate();
  if (max_ast_depth == 0) {
    throw std::invalid_argument("max_ast_depth must be positive");
  }
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open predicate JSONL: " + path);

  std::vector<PredicateRecord> records;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    if (records.size() > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("Too many predicate records: " + path);
    }
    try {
      // Metadata adds only a shallow wrapper around the predicate tree.
      JsonValue root = JsonParser(line, max_ast_depth + 8).parse_document();
      records.push_back(parse_record(root, schema,
                                     static_cast<uint32_t>(records.size()),
                                     max_ast_depth));
    } catch (const std::exception& error) {
      throw std::runtime_error(path + ":" + std::to_string(line_number) +
                               ": " + error.what());
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("Failed while reading predicate JSONL: " + path);
  }
  if (records.empty()) {
    throw std::runtime_error("Predicate JSONL contains no records: " + path);
  }
  if (expected_query_count != 0 && records.size() != expected_query_count) {
    throw std::runtime_error(
        "Predicate record count does not match query-vector rows: expected " +
        std::to_string(expected_query_count) + ", found " +
        std::to_string(records.size()));
  }
  return records;
}

}  // namespace rcfs
