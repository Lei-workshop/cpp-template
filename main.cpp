#include <string>

#include <spdlog/spdlog.h>

#include <argparse/argparse.hpp>

auto main(int argc, char* argv[]) -> int {
  argparse::ArgumentParser program("hello");
  program.add_argument("name").default_value("world").help("Your name");
  program.parse_args(argc, argv);
  spdlog::info("Hello, {}!", program.get<std::string>("name"));
}
