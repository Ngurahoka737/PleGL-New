#include "io/SettingsText.h"

#include <charconv>
#include <cmath>
#include <cstring>
#include <sstream>
#include <system_error>

namespace plegl {

namespace {

void parseFloats(const std::string& s, float* out, int n) {
  float tmp[3] = {};
  const char* p = s.data();
  const char* end = p + s.size();
  for (int i = 0; i < n; ++i) {
    while (p < end && *p == ' ') ++p;
    const auto r = std::from_chars(p, end, tmp[i]);
    if (r.ec != std::errc() || !std::isfinite(tmp[i])) return;
    p = r.ptr;
  }
  std::memcpy(out, tmp, sizeof(float) * n);
}

}  // namespace

void SettingsWriter::put(const char* key, const std::string& value) {
  text += key;
  text += ' ';
  text += value;
  text += '\n';
}

void SettingsWriter::put(const char* key, float v) {
  char buf[32];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v);  // Shortest exact form, locale free.
  put(key, std::string(buf, r.ptr));
}

void SettingsWriter::put(const char* key, int v) { put(key, std::to_string(v)); }

void SettingsWriter::put(const char* key, bool v) { put(key, std::string(v ? "1" : "0")); }

void SettingsWriter::put(const char* key, const Vec3& v) {
  std::string s;
  for (int i = 0; i < 3; ++i) {
    char buf[32];
    const auto r = std::to_chars(buf, buf + sizeof(buf), v[i]);
    if (i) s += ' ';
    s.append(buf, r.ptr);
  }
  put(key, s);
}

SettingsReader::SettingsReader(const std::string& text) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    const auto space = line.find(' ');
    if (space == std::string::npos || space == 0) continue;
    values_[line.substr(0, space)] = line.substr(space + 1);
  }
}

void SettingsReader::get(const char* key, std::string& out) const {
  if (auto it = values_.find(key); it != values_.end()) out = it->second;
}

void SettingsReader::get(const char* key, float& out) const {
  if (auto it = values_.find(key); it != values_.end()) parseFloats(it->second, &out, 1);
}

void SettingsReader::get(const char* key, Vec3& out) const {
  if (auto it = values_.find(key); it != values_.end()) parseFloats(it->second, &out.x, 3);
}

void SettingsReader::get(const char* key, bool& out) const {
  if (auto it = values_.find(key); it != values_.end()) out = it->second == "1";
}

void SettingsReader::get(const char* key, int& out) const {
  if (auto it = values_.find(key); it != values_.end()) {
    int v = 0;
    const char* s = it->second.data();
    if (std::from_chars(s, s + it->second.size(), v).ec == std::errc()) out = v;
  }
}

}  // namespace plegl
