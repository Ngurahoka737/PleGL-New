#pragma once

#include <map>
#include <string>

#include "core/Types.h"

namespace plegl {

// Settings text, as stored in a project's SETT chunk: one "key value" pair per line. Numbers are
// written in their shortest exact form and read back without the locale.
class SettingsWriter {
 public:
  std::string text;

  void put(const char* key, const std::string& value);
  void put(const char* key, float v);
  void put(const char* key, int v);
  void put(const char* key, bool v);
  void put(const char* key, const Vec3& v);
};

class SettingsReader {
 public:
  explicit SettingsReader(const std::string& text);

  // Each getter leaves `out` unchanged when the key is missing or malformed, so files from older
  // versions keep the current defaults for settings they do not have.
  void get(const char* key, std::string& out) const;
  void get(const char* key, float& out) const;
  void get(const char* key, Vec3& out) const;
  void get(const char* key, bool& out) const;
  void get(const char* key, int& out) const;
  // Values outside [0, count) leave `out` unchanged too, so a file from a newer version with an
  // enum value this build does not know keeps the default.
  template <class E>
  void getEnum(const char* key, E& out, int count) const {
    int v = static_cast<int>(out);
    get(key, v);
    if (v >= 0 && v < count) out = static_cast<E>(v);
  }

 private:
  std::map<std::string, std::string> values_;
};

}  // namespace plegl
