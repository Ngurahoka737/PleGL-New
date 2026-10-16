#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <string>

#include "io/SettingsText.h"

using namespace plegl;

namespace {

enum class Fruit { Apple, Pear, Plum };

}  // namespace

TEST_CASE("settings text round-trips every value type exactly") {
  SettingsWriter w;
  w.put("text", std::string("a b  c"));
  w.put("float", 0.1f);
  w.put("tiny", std::numeric_limits<float>::denorm_min());
  w.put("negative", -1234.5678f);
  w.put("int", -42);
  w.put("on", true);
  w.put("off", false);
  w.put("vec", Vec3{1.0f / 3.0f, -0.0f, 1e30f});
  w.put("fruit", 2);

  const SettingsReader r(w.text);
  std::string text;
  float f = 0, tiny = 0, negative = 0;
  int i = 0;
  bool on = false, off = true;
  Vec3 vec{0.0f};
  Fruit fruit = Fruit::Apple;
  r.get("text", text);
  r.get("float", f);
  r.get("tiny", tiny);
  r.get("negative", negative);
  r.get("int", i);
  r.get("on", on);
  r.get("off", off);
  r.get("vec", vec);
  r.getEnum("fruit", fruit, 3);
  CHECK(text == "a b  c");
  CHECK(f == 0.1f);
  CHECK(tiny == std::numeric_limits<float>::denorm_min());
  CHECK(negative == -1234.5678f);
  CHECK(i == -42);
  CHECK(on);
  CHECK_FALSE(off);
  CHECK(vec.x == 1.0f / 3.0f);
  CHECK(std::signbit(vec.y));
  CHECK(vec.z == 1e30f);
  CHECK(fruit == Fruit::Plum);
}

TEST_CASE("missing, malformed and out-of-range values keep the current value") {
  const SettingsReader r(
      "float abc\n"
      "nan nan\n"
      "inf inf\n"
      "int 12x\n"
      "big 99999999999\n"
      "vec 1 2\n"
      "fruit 3\n"
      "negative_fruit -1\n"
      " leading space\n"
      "nospace\n");
  float f = 7.0f, nan = 7.0f, inf = 7.0f;
  int i = 5, big = 5, leading = 5;
  Vec3 vec{9.0f};
  Fruit fruit = Fruit::Pear, negative = Fruit::Pear;
  std::string missing = "kept", nospace = "kept";
  r.get("float", f);
  r.get("nan", nan);
  r.get("inf", inf);
  r.get("int", i);
  r.get("big", big);
  r.get("vec", vec);
  r.getEnum("fruit", fruit, 3);
  r.getEnum("negative_fruit", negative, 3);
  r.get("missing", missing);
  r.get("nospace", nospace);
  r.get("", leading);
  CHECK(f == 7.0f);
  CHECK(nan == 7.0f);
  CHECK(inf == 7.0f);
  CHECK(i == 12);  // Leading digits are a number, as in every version so far.
  CHECK(big == 5);
  CHECK(vec == Vec3{9.0f});
  CHECK(fruit == Fruit::Pear);
  CHECK(negative == Fruit::Pear);
  CHECK(missing == "kept");
  CHECK(nospace == "kept");
  CHECK(leading == 5);
}

TEST_CASE("the last copy of a key wins and other lines do not matter") {
  const SettingsReader r("a 1\nunknown.key whatever\r\na 2\n\n");
  int a = 0;
  r.get("a", a);
  CHECK(a == 2);
  std::string unknown;
  r.get("unknown.key", unknown);
  CHECK(unknown == "whatever\r");  // Stored as written; the app writes "\n" only.
}
