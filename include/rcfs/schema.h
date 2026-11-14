#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "rcfs/types.h"

namespace rcfs {

struct Schema {
  std::vector<std::string> numeric_names;
  std::vector<std::string> categorical_names;
  std::vector<std::vector<std::string>> categorical_value_names;
  std::vector<CategoryCode> category_code_offsets;
  CategoryCode total_category_codes = 0;

  size_t attribute_count() const { return numeric_names.size() + categorical_names.size(); }
  AttributeId unified_numeric(NumericAttrId id) const {
    if (id >= numeric_names.size()) throw std::out_of_range("Numeric attribute ID");
    return id;
  }
  AttributeId unified_categorical(CategoricalAttrId id) const {
    if (id >= categorical_names.size()) throw std::out_of_range("Categorical attribute ID");
    return static_cast<AttributeId>(numeric_names.size() + id);
  }
  AttributeType attribute_type(AttributeId id) const {
    if (id >= attribute_count()) throw std::out_of_range("Unified attribute ID");
    return id < numeric_names.size() ? AttributeType::kNumeric : AttributeType::kCategorical;
  }
  const std::string& attribute_name(AttributeId id) const {
    return attribute_type(id) == AttributeType::kNumeric ? numeric_names[id]
        : categorical_names[id - numeric_names.size()];
  }
  AttributeId attribute_id(const std::string& name) const {
    for (AttributeId id = 0; id < attribute_count(); ++id)
      if (attribute_name(id) == name) return id;
    throw std::invalid_argument("Unknown attribute: " + name);
  }

  NumericAttrId numeric_id(const std::string& name) const {
    const auto it = std::find(numeric_names.begin(), numeric_names.end(), name);
    if (it == numeric_names.end()) {
      throw std::invalid_argument("Unknown numeric attribute: " + name);
    }
    return static_cast<NumericAttrId>(it - numeric_names.begin());
  }

  CategoricalAttrId categorical_id(const std::string& name) const {
    const auto it =
        std::find(categorical_names.begin(), categorical_names.end(), name);
    if (it == categorical_names.end()) {
      throw std::invalid_argument("Unknown categorical attribute: " + name);
    }
    return static_cast<CategoricalAttrId>(it - categorical_names.begin());
  }

  ValueId value_id(CategoricalAttrId attribute,
                   const std::string& value_name) const {
    if (attribute >= categorical_value_names.size()) {
      throw std::out_of_range("Categorical attribute ID is out of range");
    }
    const auto& names = categorical_value_names[attribute];
    const auto it = std::find(names.begin(), names.end(), value_name);
    if (it == names.end()) {
      throw std::invalid_argument("Unknown value '" + value_name +
                                  "' for categorical attribute " +
                                  categorical_names[attribute]);
    }
    return static_cast<ValueId>(it - names.begin());
  }

  CategoryCode category_code(CategoricalAttrId attribute,
                             ValueId value) const {
    if (attribute >= categorical_value_names.size() ||
        attribute >= category_code_offsets.size() ||
        value >= categorical_value_names[attribute].size()) {
      throw std::out_of_range("Categorical code components are out of range");
    }
    return category_code_offsets[attribute] + value;
  }

  void validate() const {
    if (categorical_names.size() != categorical_value_names.size() ||
        categorical_names.size() != category_code_offsets.size()) {
      throw std::runtime_error("Schema categorical arrays have different sizes");
    }
    std::vector<std::string> all_names = numeric_names;
    all_names.insert(all_names.end(), categorical_names.begin(), categorical_names.end());
    std::sort(all_names.begin(), all_names.end());
    if (std::find(all_names.begin(), all_names.end(), "") != all_names.end() ||
        std::adjacent_find(all_names.begin(), all_names.end()) != all_names.end())
      throw std::runtime_error("Attribute names must be nonempty and unique");
    uint64_t next = 0;
    for (size_t i = 0; i < categorical_names.size(); ++i) {
      if (categorical_names[i].empty() || categorical_value_names[i].empty()) {
        throw std::runtime_error("Categorical schema contains an empty field");
      }
      if (category_code_offsets[i] != next) {
        throw std::runtime_error("Categorical code offsets are not contiguous");
      }
      auto names = categorical_value_names[i];
      std::sort(names.begin(), names.end());
      if (std::adjacent_find(names.begin(), names.end()) != names.end())
        throw std::runtime_error("Duplicate categorical dictionary value");
      next += categorical_value_names[i].size();
      if (next >= kMissingValue) throw std::overflow_error("Too many category codes");
    }
    if (next != total_category_codes) {
      throw std::runtime_error("Schema total category code count is inconsistent");
    }
  }
};

inline Schema make_schema(const Dataset& dataset) {
  Schema schema;
  for (const auto& column : dataset.numeric_columns) {
    schema.numeric_names.push_back(column.name);
  }
  CategoryCode next_code = 0;
  for (const auto& column : dataset.categorical_columns) {
    schema.categorical_names.push_back(column.name);
    schema.categorical_value_names.push_back(column.value_names);
    schema.category_code_offsets.push_back(next_code);
    next_code += static_cast<CategoryCode>(column.value_names.size());
  }
  schema.total_category_codes = next_code;
  schema.validate();
  return schema;
}

}  // namespace rcfs
