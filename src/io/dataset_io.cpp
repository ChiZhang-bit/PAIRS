#include "rcfs/dataset_io.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <map>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace rcfs {
namespace {

constexpr std::array<char, 8> kAttributeMagic = {'U', 'R', 'F', 'A',
                                                  'T', 'T', 'R', '\0'};
constexpr uint32_t kAttributeVersion = 1;
constexpr uint32_t kAttributeHeaderBytes = 32;
constexpr std::array<const char*, 8> kDefaultColors = {
    "blue", "orange", "red", "green",
    "yellow", "purple", "black", "white"};

bool is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

void require_little_endian() {
  if (!is_little_endian()) {
    throw std::runtime_error(
        "RCFS portable binaries currently require a little-endian host");
  }
}

template <typename T>
T read_pod(std::istream& input, const std::string& path) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input) {
    throw std::runtime_error("Unexpected end of file: " + path);
  }
  return value;
}

template <typename T>
void read_array(std::istream& input, std::vector<T>* output, size_t count,
                const std::string& path) {
  output->resize(count);
  if (count == 0) return;
  if (count > static_cast<size_t>(
                  std::numeric_limits<std::streamsize>::max()) /
                  sizeof(T)) {
    throw std::runtime_error("Binary array is too large to read: " + path);
  }
  input.read(reinterpret_cast<char*>(output->data()),
             static_cast<std::streamsize>(count * sizeof(T)));
  if (!input) {
    throw std::runtime_error("Unexpected end of file: " + path);
  }
}

template <typename T>
void write_pod(std::ostream& output, const T& value,
               const std::string& path) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!output) throw std::runtime_error("Failed to write file: " + path);
}

template <typename T>
void write_array(std::ostream& output, const std::vector<T>& values,
                 const std::string& path) {
  if (values.empty()) return;
  if (values.size() > static_cast<size_t>(
                          std::numeric_limits<std::streamsize>::max()) /
                          sizeof(T)) {
    throw std::runtime_error("Binary array is too large to write: " + path);
  }
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!output) throw std::runtime_error("Failed to write file: " + path);
}

void ensure_no_trailing_bytes(std::istream& input, const std::string& path) {
  char byte = 0;
  if (input.read(&byte, 1)) {
    throw std::runtime_error("Unexpected trailing bytes: " + path);
  }
}

uint64_t file_size(std::ifstream* input, const std::string& path) {
  input->seekg(0, std::ios::end);
  const std::streampos end = input->tellg();
  if (end < 0) throw std::runtime_error("Cannot determine file size: " + path);
  input->seekg(0, std::ios::beg);
  if (!*input) throw std::runtime_error("Cannot seek file: " + path);
  return static_cast<uint64_t>(end);
}

size_t checked_element_count(uint64_t rows, uint64_t cols,
                             const std::string& path) {
  if (rows == 0 || cols == 0) {
    throw std::runtime_error("Binary matrix has zero rows or columns: " + path);
  }
  if (rows > std::numeric_limits<size_t>::max() / cols) {
    throw std::runtime_error("Binary matrix element count overflows: " + path);
  }
  return static_cast<size_t>(rows * cols);
}

void skip_space(const std::string& text, size_t* cursor) {
  while (*cursor < text.size() &&
         std::isspace(static_cast<unsigned char>(text[*cursor]))) {
    ++*cursor;
  }
}

std::string parse_mapping_string(const std::string& text, size_t* cursor,
                                 const std::string& path) {
  skip_space(text, cursor);
  if (*cursor >= text.size() || text[*cursor] != '"') {
    throw std::runtime_error("Expected JSON string in color mapping: " + path);
  }
  ++*cursor;
  std::string value;
  while (*cursor < text.size()) {
    const char ch = text[(*cursor)++];
    if (ch == '"') return value;
    if (static_cast<unsigned char>(ch) < 0x20) {
      throw std::runtime_error("Control byte in color mapping: " + path);
    }
    if (ch != '\\') {
      value.push_back(ch);
      continue;
    }
    if (*cursor >= text.size()) {
      throw std::runtime_error("Truncated escape in color mapping: " + path);
    }
    const char escaped = text[(*cursor)++];
    switch (escaped) {
      case '"':
      case '\\':
      case '/':
        value.push_back(escaped);
        break;
      case 'b':
        value.push_back('\b');
        break;
      case 'f':
        value.push_back('\f');
        break;
      case 'n':
        value.push_back('\n');
        break;
      case 'r':
        value.push_back('\r');
        break;
      case 't':
        value.push_back('\t');
        break;
      default:
        throw std::runtime_error(
            "Unsupported escape in color mapping JSON: " + path);
    }
  }
  throw std::runtime_error("Unterminated string in color mapping: " + path);
}

std::vector<std::string> read_color_names(const std::string& path,
                                          uint32_t color_count) {
  if (path.empty()) {
    if (color_count != kDefaultColors.size()) {
      throw std::runtime_error(
          "A color mapping is required when color_count is not 8");
    }
    return std::vector<std::string>(kDefaultColors.begin(),
                                    kDefaultColors.end());
  }

  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open color mapping: " + path);
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  size_t cursor = 0;
  skip_space(text, &cursor);
  if (cursor >= text.size() || text[cursor++] != '{') {
    throw std::runtime_error("Color mapping must be a JSON object: " + path);
  }

  std::vector<std::string> names(color_count);
  std::vector<bool> present(color_count, false);
  skip_space(text, &cursor);
  if (cursor < text.size() && text[cursor] == '}') ++cursor;
  while (cursor < text.size() && text[cursor - 1] != '}') {
    const std::string key = parse_mapping_string(text, &cursor, path);
    skip_space(text, &cursor);
    if (cursor >= text.size() || text[cursor++] != ':') {
      throw std::runtime_error("Missing ':' in color mapping: " + path);
    }
    const std::string name = parse_mapping_string(text, &cursor, path);
    size_t parsed = 0;
    unsigned long numeric_key = 0;
    try {
      numeric_key = std::stoul(key, &parsed, 10);
    } catch (const std::exception&) {
      throw std::runtime_error("Non-numeric color code in mapping: " + path);
    }
    if (parsed != key.size() || numeric_key >= color_count ||
        present[numeric_key]) {
      throw std::runtime_error("Invalid or duplicate color code in mapping: " +
                               path);
    }
    names[numeric_key] = name;
    present[numeric_key] = true;
    skip_space(text, &cursor);
    if (cursor >= text.size()) {
      throw std::runtime_error("Truncated color mapping object: " + path);
    }
    if (text[cursor] == '}') {
      ++cursor;
      break;
    }
    if (text[cursor++] != ',') {
      throw std::runtime_error("Expected ',' in color mapping: " + path);
    }
    skip_space(text, &cursor);
  }
  skip_space(text, &cursor);
  if (cursor != text.size() ||
      std::find(present.begin(), present.end(), false) != present.end()) {
    throw std::runtime_error("Incomplete or malformed color mapping: " + path);
  }
  std::unordered_set<std::string> unique_names(names.begin(), names.end());
  if (unique_names.size() != names.size()) {
    throw std::runtime_error("Color mapping contains duplicate names: " + path);
  }
  return names;
}


// Field-scoped dictionaries use contiguous codes: {field: {"0": "label", ...}}.
std::map<std::string, std::vector<std::string>> parse_tag_names(
    const std::string& text, const std::string& path) {
  for (size_t i = 0; i < text.size();) {
    const auto first = static_cast<unsigned char>(text[i++]);
    if (first < 0x80) continue;
    uint32_t codepoint = 0, minimum = 0;
    size_t continuation = 0;
    if (first >= 0xc2 && first <= 0xdf) {
      codepoint = first & 0x1f; minimum = 0x80; continuation = 1;
    } else if (first >= 0xe0 && first <= 0xef) {
      codepoint = first & 0x0f; minimum = 0x800; continuation = 2;
    } else if (first >= 0xf0 && first <= 0xf4) {
      codepoint = first & 0x07; minimum = 0x10000; continuation = 3;
    } else {
      throw std::runtime_error("Invalid UTF-8 in tag mapping: " + path);
    }
    if (continuation > text.size() - i)
      throw std::runtime_error("Truncated UTF-8 in tag mapping: " + path);
    while (continuation--) {
      const auto byte = static_cast<unsigned char>(text[i++]);
      if ((byte & 0xc0) != 0x80)
        throw std::runtime_error("Invalid UTF-8 in tag mapping: " + path);
      codepoint = (codepoint << 6) | (byte & 0x3f);
    }
    if (codepoint < minimum || codepoint > 0x10ffff ||
        (codepoint >= 0xd800 && codepoint <= 0xdfff))
      throw std::runtime_error("Invalid UTF-8 in tag mapping: " + path);
  }
  size_t cursor = 0;
  auto expect = [&](char ch) {
    skip_space(text, &cursor);
    if (cursor >= text.size() || text[cursor++] != ch)
      throw std::runtime_error("Malformed tag mapping: " + path);
  };
  auto consume = [&](char ch) {
    skip_space(text, &cursor);
    if (cursor < text.size() && text[cursor] == ch) { ++cursor; return true; }
    return false;
  };
  std::map<std::string, std::vector<std::string>> result;
  expect('{');
  if (consume('}')) throw std::runtime_error("Empty tag mapping");
  do {
    const auto field = parse_mapping_string(text, &cursor, path);
    if (result.count(field)) throw std::runtime_error("Duplicate tag mapping field");
    expect(':'); expect('{');
    std::map<uint32_t, std::string> codes;
    if (consume('}')) throw std::runtime_error("Empty tag dictionary");
    do {
      const auto key = parse_mapping_string(text, &cursor, path);
      if (key.empty() || key.size() > 5 || (key.size()>1 && key[0]=='0') ||
          key.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid tag mapping code");
      const auto code = static_cast<uint32_t>(std::stoul(key));
      expect(':'); const auto label = parse_mapping_string(text, &cursor, path);
      if (code > 65535 || label.empty() || !codes.emplace(code,label).second)
        throw std::runtime_error("Invalid/duplicate tag mapping entry");
    } while (consume(','));
    expect('}');
    std::vector<std::string> names;
    std::unordered_set<std::string> unique;
    for (const auto& entry : codes) {
      if (entry.first != names.size() || !unique.insert(entry.second).second)
        throw std::runtime_error("Noncontiguous codes or duplicate tag labels");
      names.push_back(entry.second);
    }
    result.emplace(field, std::move(names));
  } while (consume(','));
  expect('}'); skip_space(text,&cursor);
  if (cursor != text.size()) throw std::runtime_error("Trailing tag mapping content");
  return result;
}

std::map<std::string, std::vector<std::string>> read_tag_names(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open tag mapping: " + path);
  const std::string text((std::istreambuf_iterator<char>(input)), {});
  return parse_tag_names(text, path);
}

AttributeTable read_attributes_v2(std::ifstream& input, const std::string& path,
    const std::string& mapping_path, uint32_t header_bytes, uint64_t rows,
    uint32_t nn, uint32_t nc, uint32_t version) {
  if (!rows || rows > std::numeric_limits<uint32_t>::max() || !nn || !nc ||
      nn > 4096 || nc > 4096 || nn+nc > 4096 || header_bytes != 32+24*(nn+nc))
    throw std::runtime_error("Invalid v2 attribute header");
  struct Descriptor { std::string name; uint8_t kind, type; uint32_t size; };
  std::vector<Descriptor> descriptors;
  std::unordered_set<std::string> unique;
  uint64_t row_bytes=0;
  for (uint32_t i=0; i<nn+nc; ++i) {
    std::array<char,16> raw{};
    input.read(raw.data(),raw.size());
    auto end=std::find(raw.begin(),raw.end(),'\0');
    if (!input || end==raw.begin() || end==raw.end() ||
        std::any_of(end,raw.end(),[](char c){return c!=0;}))
      throw std::runtime_error("Invalid v2 descriptor name");
    Descriptor d; d.name.assign(raw.begin(),end);
    if (std::any_of(d.name.begin(),d.name.end(),[](unsigned char c){return c>127;}) ||
        !unique.insert(d.name).second) throw std::runtime_error("Invalid/duplicate v2 field");
    d.kind=read_pod<uint8_t>(input,path); d.type=read_pod<uint8_t>(input,path);
    const auto reserved=read_pod<uint16_t>(input,path);
    d.size=read_pod<uint32_t>(input,path);
    if (reserved || d.kind!=(i<nn?1:2) || (version==2 && d.type==4) ||
        (d.kind==1 && (d.size || (d.type!=1 && d.type!=2))) ||
        (d.kind==2 && ((d.type!=3 && d.type!=4) || !d.size ||
                      d.size>(d.type==3?256u:65536u))))
      throw std::runtime_error("Invalid v2 field descriptor");
    row_bytes+=d.type==3?1:(d.type==4?2:4); descriptors.push_back(std::move(d));
  }
  const uint64_t payload_end = header_bytes + rows * row_bytes;
  const uint64_t bytes = std::filesystem::file_size(path);
  std::map<std::string, std::vector<std::string>> dictionaries;
  if (version == 4) {
    if (!mapping_path.empty())
      throw std::runtime_error("v4 attributes embed their dictionary; omit external mapping options");
    // Validate and parse the dictionary before allocating the attribute columns.
    // The fixed-width payload is unchanged from v2/v3.
    if (bytes < payload_end + sizeof(uint32_t))
      throw std::runtime_error("Truncated embedded attribute mapping");
    input.seekg(static_cast<std::streamoff>(payload_end), std::ios::beg);
    const uint32_t mapping_bytes = read_pod<uint32_t>(input, path);
    if (!mapping_bytes || bytes - payload_end - sizeof(uint32_t) != mapping_bytes)
      throw std::runtime_error("Embedded attribute mapping length mismatch");
    std::string text(mapping_bytes, '\0');
    input.read(text.data(), static_cast<std::streamsize>(mapping_bytes));
    if (!input) throw std::runtime_error("Truncated embedded attribute mapping");
    dictionaries = parse_tag_names(text, path + " (embedded mapping)");
    ensure_no_trailing_bytes(input, path);
    input.clear();
    input.seekg(header_bytes, std::ios::beg);
    if (!input) throw std::runtime_error("Cannot seek attribute payload: " + path);
  } else {
    if (bytes != payload_end) throw std::runtime_error("Attribute payload length mismatch");
    dictionaries = read_tag_names(mapping_path.empty()
        ? (std::filesystem::path(path).parent_path()/"tag_mapping.json").string() : mapping_path);
  }
  if (dictionaries.size()!=nc) throw std::runtime_error("Tag mapping field count mismatch");
  AttributeTable table; table.rows=rows;
  uint32_t offset=0;
  // The payload stores whole columns in descriptor order.
  for (const auto& d:descriptors) {
    if (d.kind==1) {
      NumericColumn column; column.name=d.name;
      if (d.type==1) read_array(input,&column.values,rows,path);
      else {
        std::vector<int32_t> values; read_array(input,&values,rows,path);
        column.values.reserve(rows);
        for (auto value:values) {
          // Keep integer keys exact when converting to the index's float32 storage.
          if (value < -16777216 || value > 16777216)
            throw std::runtime_error("Integer not exactly representable by RCFS float32");
          column.values.push_back(static_cast<float>(value));
        }
      }
      for (float value:column.values) if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite v2 numeric value");
      table.numeric_columns.push_back(std::move(column));
    } else {
      auto it=dictionaries.find(d.name);
      if (it==dictionaries.end() || it->second.size()!=d.size)
        throw std::runtime_error("v2 dictionary/schema mismatch");
      CategoricalColumn column; column.name=d.name; column.value_names=it->second;
      column.code_offset=offset; offset+=d.size;
      column.values.reserve(rows);
      const auto append_codes=[&](const auto& values) {
        for (auto code:values) {
          if (code>=d.size) throw std::runtime_error("v2 category code outside dictionary");
          column.values.push_back(code);
        }
      };
      if (d.type==3) {
        std::vector<uint8_t> values; read_array(input,&values,rows,path);
        append_codes(values);
      } else {
        std::vector<uint16_t> values; read_array(input,&values,rows,path);
        append_codes(values);
      }
      table.categorical_columns.push_back(std::move(column));
    }
  }
  if (version != 4) ensure_no_trailing_bytes(input,path);
  return table;
}

}  // namespace

VectorFileFormat parse_vector_file_format(const std::string& name) {
  if (name == "bin" || name == "dense" || name == "dense_bin") {
    return VectorFileFormat::kDenseFloatBinary;
  }
  if (name == "fvecs") return VectorFileFormat::kFvecs;
  throw std::invalid_argument("Unsupported vector file format: " + name);
}

const char* vector_file_format_name(VectorFileFormat format) {
  switch (format) {
    case VectorFileFormat::kDenseFloatBinary:
      return "bin";
    case VectorFileFormat::kFvecs:
      return "fvecs";
  }
  return "unknown";
}

DenseFloatMatrix read_dense_float_bin(const std::string& path) {
  require_little_endian();
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open dense binary: " + path);
  DenseFloatMatrix matrix;
  matrix.rows = read_pod<uint32_t>(input, path);
  matrix.cols = read_pod<uint32_t>(input, path);
  const size_t count = checked_element_count(matrix.rows, matrix.cols, path);
  read_array(input, &matrix.values, count, path);
  ensure_no_trailing_bytes(input, path);
  return matrix;
}

DenseFloatMatrix read_fvecs(const std::string& path) {
  require_little_endian();
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open fvecs file: " + path);
  const uint64_t bytes = file_size(&input, path);
  if (bytes < sizeof(int32_t)) {
    throw std::runtime_error("Empty or truncated fvecs file: " + path);
  }
  const int32_t signed_dim = read_pod<int32_t>(input, path);
  if (signed_dim <= 0) {
    throw std::runtime_error("Invalid fvecs dimension: " + path);
  }
  const uint64_t dimension = static_cast<uint32_t>(signed_dim);
  const uint64_t record_bytes = sizeof(int32_t) + dimension * sizeof(float);
  if (bytes % record_bytes != 0) {
    throw std::runtime_error("Truncated fvecs record: " + path);
  }
  const uint64_t row_count = bytes / record_bytes;
  if (row_count == 0 || row_count > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Unsupported fvecs row count: " + path);
  }

  DenseFloatMatrix matrix;
  matrix.rows = static_cast<uint32_t>(row_count);
  matrix.cols = static_cast<uint32_t>(dimension);
  matrix.values.resize(checked_element_count(row_count, dimension, path));
  input.clear();
  input.seekg(0, std::ios::beg);
  for (uint32_t row = 0; row < matrix.rows; ++row) {
    const int32_t row_dim = read_pod<int32_t>(input, path);
    if (row_dim != signed_dim) {
      throw std::runtime_error("Inconsistent fvecs dimension at row " +
                               std::to_string(row) + ": " + path);
    }
    input.read(
        reinterpret_cast<char*>(matrix.values.data() +
                                static_cast<size_t>(row) * matrix.cols),
        static_cast<std::streamsize>(matrix.cols * sizeof(float)));
    if (!input) {
      throw std::runtime_error("Truncated fvecs row " + std::to_string(row) +
                               ": " + path);
    }
  }
  ensure_no_trailing_bytes(input, path);
  return matrix;
}

DenseFloatMatrix read_vectors(const std::string& path,
                              VectorFileFormat format) {
  switch (format) {
    case VectorFileFormat::kDenseFloatBinary:
      return read_dense_float_bin(path);
    case VectorFileFormat::kFvecs:
      return read_fvecs(path);
  }
  throw std::invalid_argument("Unknown vector file format enum");
}

IdMatrix read_ivecs(const std::string& path) {
  require_little_endian();
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open ivecs file: " + path);
  const uint64_t bytes = file_size(&input, path);
  if (bytes < sizeof(int32_t)) {
    throw std::runtime_error("Empty or truncated ivecs file: " + path);
  }
  const int32_t signed_k = read_pod<int32_t>(input, path);
  if (signed_k <= 0) throw std::runtime_error("Invalid ivecs K: " + path);
  const uint64_t record_bytes =
      sizeof(int32_t) * (static_cast<uint64_t>(signed_k) + 1);
  if (bytes % record_bytes != 0) {
    throw std::runtime_error("Truncated ivecs record: " + path);
  }
  const uint64_t row_count = bytes / record_bytes;
  if (row_count == 0 || row_count > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Unsupported ivecs row count: " + path);
  }
  IdMatrix matrix;
  matrix.rows = static_cast<uint32_t>(row_count);
  matrix.cols = static_cast<uint32_t>(signed_k);
  matrix.ids.resize(checked_element_count(row_count, matrix.cols, path));
  input.clear();
  input.seekg(0, std::ios::beg);
  for (uint32_t row = 0; row < matrix.rows; ++row) {
    const int32_t row_k = read_pod<int32_t>(input, path);
    if (row_k != signed_k) {
      throw std::runtime_error("Inconsistent ivecs K at row " +
                               std::to_string(row) + ": " + path);
    }
    for (uint32_t col = 0; col < matrix.cols; ++col) {
      const int32_t id = read_pod<int32_t>(input, path);
      if (id < -1) {
        throw std::runtime_error("Invalid negative ivecs ID at row " +
                                 std::to_string(row) + ": " + path);
      }
      matrix.ids[static_cast<size_t>(row) * matrix.cols + col] =
          id == -1 ? kInvalidNodeId : static_cast<NodeId>(id);
    }
  }
  ensure_no_trailing_bytes(input, path);
  return matrix;
}

ResultMatrix read_rcfs_result(const std::string& path) {
  require_little_endian();
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open RCFS result: " + path);
  ResultMatrix result;
  result.rows = read_pod<uint32_t>(input, path);
  result.k = read_pod<uint32_t>(input, path);
  const size_t count = checked_element_count(result.rows, result.k, path);
  read_array(input, &result.ids, count, path);
  read_array(input, &result.distances, count, path);
  ensure_no_trailing_bytes(input, path);
  return result;
}

void write_rcfs_result(const std::string& path, const ResultMatrix& result) {
  require_little_endian();
  const size_t count = checked_element_count(result.rows, result.k, path);
  if (result.ids.size() != count || result.distances.size() != count) {
    throw std::invalid_argument("RCFS result shape does not match its payload");
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("Cannot create RCFS result: " + path);
  write_pod(output, result.rows, path);
  write_pod(output, result.k, path);
  write_array(output, result.ids, path);
  write_array(output, result.distances, path);
}

AttributeTable read_sift_gist_base_attributes(
    const std::string& path, const std::string& color_mapping_path) {
  require_little_endian();
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open SIFT/GIST base attributes: " + path);
  }
  std::array<char, 8> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != kAttributeMagic) {
    throw std::runtime_error("Invalid base-attribute magic: " + path);
  }
  const uint32_t version = read_pod<uint32_t>(input, path);
  const uint32_t header_bytes = read_pod<uint32_t>(input, path);
  const uint64_t rows = read_pod<uint64_t>(input, path);
  const uint32_t color_count = read_pod<uint32_t>(input, path);
  const uint32_t flags = read_pod<uint32_t>(input, path);
  // Versions 2-4 use these header slots for numeric and categorical field counts.
  if (version == 2 || version == 3 || version == 4) return read_attributes_v2(input, path, color_mapping_path,
      header_bytes, rows, color_count, flags, version);
  if (version != kAttributeVersion || header_bytes != kAttributeHeaderBytes ||
      flags != 0) {
    throw std::runtime_error("Unsupported base-attribute header: " + path);
  }
  if (rows == 0 || rows > std::numeric_limits<uint32_t>::max() ||
      color_count == 0 || color_count > 256) {
    throw std::runtime_error("Invalid base-attribute shape: " + path);
  }
  const size_t count = static_cast<size_t>(rows);
  std::vector<uint8_t> bool_values;
  std::vector<float> range1_values;
  std::vector<int32_t> range2_values;
  std::vector<uint8_t> color_values;
  read_array(input, &bool_values, count, path);
  read_array(input, &range1_values, count, path);
  read_array(input, &range2_values, count, path);
  read_array(input, &color_values, count, path);
  ensure_no_trailing_bytes(input, path);

  AttributeTable table;
  table.rows = rows;
  table.numeric_columns.resize(2);
  table.numeric_columns[0].name = "range1";
  table.numeric_columns[0].values = std::move(range1_values);
  table.numeric_columns[1].name = "range2";
  table.numeric_columns[1].values.resize(count);
  std::transform(range2_values.begin(), range2_values.end(),
                 table.numeric_columns[1].values.begin(),
                 [](int32_t value) { return static_cast<float>(value); });

  table.categorical_columns.resize(2);
  CategoricalColumn& bool_column = table.categorical_columns[0];
  bool_column.name = "bool";
  bool_column.value_names = {"false", "true"};
  bool_column.code_offset = 0;
  bool_column.values.resize(count);
  CategoricalColumn& color_column = table.categorical_columns[1];
  color_column.name = "color";
  color_column.value_names = read_color_names(color_mapping_path, color_count);
  color_column.code_offset = 2;
  color_column.values.resize(count);

  for (size_t row = 0; row < count; ++row) {
    if (bool_values[row] > 1) {
      throw std::runtime_error("Invalid bool value at row " +
                               std::to_string(row) + ": " + path);
    }
    if (!std::isfinite(table.numeric_columns[0].values[row])) {
      throw std::runtime_error("Non-finite range1 value at row " +
                               std::to_string(row) + ": " + path);
    }
    if (color_values[row] >= color_count) {
      throw std::runtime_error("Invalid color value at row " +
                               std::to_string(row) + ": " + path);
    }
    bool_column.values[row] = static_cast<ValueId>(bool_values[row]);
    color_column.values[row] = static_cast<ValueId>(color_values[row]);
  }
  return table;
}

Dataset read_sift_gist_dataset(const DatasetFiles& files) {
  if (files.vectors.empty() || files.base_attributes.empty()) {
    throw std::invalid_argument(
        "SIFT/GIST dataset requires vectors and base_attributes paths");
  }
  Dataset dataset;
  dataset.vectors = read_vectors(files.vectors, files.vector_format);
  AttributeTable attributes = read_sift_gist_base_attributes(
      files.base_attributes, files.tag_mapping.empty() ? files.color_mapping : files.tag_mapping);
  if (attributes.rows != dataset.vectors.rows) {
    throw std::runtime_error(
        "base_attrs.bin row count does not match base vectors");
  }
  dataset.numeric_columns = std::move(attributes.numeric_columns);
  dataset.categorical_columns = std::move(attributes.categorical_columns);
  make_schema(dataset).validate();
  return dataset;
}

QueryWorkload read_query_workload(const QueryFiles& files,
                                  const Schema& schema,
                                  uint32_t max_ast_depth) {
  if (files.vectors.empty() || files.predicates.empty()) {
    throw std::invalid_argument(
        "Query workload requires vectors and predicates paths");
  }
  QueryWorkload workload;
  workload.vectors = read_vectors(files.vectors, files.vector_format);
  workload.predicates = read_predicate_jsonl(
      files.predicates, schema, workload.vectors.rows, max_ast_depth);
  return workload;
}

}  // namespace rcfs
