#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "App.h"

int main(int argc, char** argv) {
  plegl::App app;
  std::string error;
  if (!app.init(&error)) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "%s", error.c_str());
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "PleGL Sculpt", error.c_str(), nullptr);
    app.shutdown();
    return 1;
  }
  // Files passed on the command line (or dropped on the executable): projects open, meshes import.
  for (int i = 1; i < argc; ++i) app.openFile(std::filesystem::path(reinterpret_cast<const char8_t*>(argv[i])));
  app.run();
  app.shutdown();
  return 0;
}
