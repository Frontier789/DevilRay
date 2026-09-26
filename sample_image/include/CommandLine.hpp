// AI-generated command line parsing (Claude), reviewed by hand before committing.

#pragma once

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

struct CommandLineOptions
{
    std::string image_path;
    int point_count = 1300;
    bool plot_points = false;
    bool help_requested = false;
};

class CommandLineError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

inline void printHelp(std::string_view program_name, std::ostream &out = std::cout)
{
    const auto defaults = CommandLineOptions{};

    out << "Usage: " << program_name << " [options] <image_path>\n"
        << "\n"
        << "Samples points from the image proportionally to its luminance and writes\n"
        << "voronoi.png and sample_points.png into the working directory.\n"
        << "\n"
        << "Arguments:\n"
        << "  <image_path>            Image file to sample\n"
        << "\n"
        << "Options:\n"
        << "  -n, --point-count <N>   Number of points to sample (default: " << defaults.point_count << ")\n"
        << "  -p, --plot-points       Draw the sampled points onto voronoi.png\n"
        << "  -h, --help              Show this help and exit\n";
}

namespace command_line_detail
{
    inline bool matchesOption(std::string_view argument, std::string_view short_name, std::string_view long_name)
    {
        return argument == short_name || argument == long_name;
    }

    inline bool looksLikeOption(std::string_view argument)
    {
        return argument.size() > 1 && argument.front() == '-';
    }

    inline int parsePositiveInteger(std::string_view text, std::string_view option_name)
    {
        const auto text_end = text.data() + text.size();

        int value = 0;
        const auto [parse_end, parse_error] = std::from_chars(text.data(), text_end, value);

        const bool is_whole_text_a_number = parse_error == std::errc{} && parse_end == text_end;
        if (!is_whole_text_a_number || value <= 0)
            throw CommandLineError(std::string(option_name) + " expects a positive integer, got '" + std::string(text) + "'");

        return value;
    }

    inline std::string_view takeOptionValue(std::vector<std::string_view>::const_iterator &argument,
                                            std::vector<std::string_view>::const_iterator arguments_end)
    {
        const auto option_name = *argument;
        const bool has_value = std::next(argument) != arguments_end;
        if (!has_value)
            throw CommandLineError(std::string(option_name) + " expects a value");

        return *++argument;
    }

    inline void requireExistingFile(const std::string &path)
    {
        if (!std::filesystem::is_regular_file(path))
            throw CommandLineError("Image file not found: '" + path + "'");
    }

    inline std::string programNameFrom(int argc, char *argv[])
    {
        const bool has_program_name = argc > 0 && argv[0] != nullptr;
        if (!has_program_name)
            return "sample_image";

        return std::filesystem::path(argv[0]).filename().string();
    }
}

inline CommandLineOptions parseCommandLine(int argc, char *argv[])
{
    using namespace command_line_detail;

    const auto arguments = std::vector<std::string_view>(argv + 1, argv + argc);

    CommandLineOptions options;
    std::optional<std::string> image_path;

    for (auto argument = arguments.begin(); argument != arguments.end(); ++argument)
    {
        if (matchesOption(*argument, "-h", "--help"))
        {
            options.help_requested = true;
            return options;
        }
        else if (matchesOption(*argument, "-p", "--plot-points"))
        {
            options.plot_points = true;
        }
        else if (matchesOption(*argument, "-n", "--point-count"))
        {
            const auto option_name = *argument;
            options.point_count = parsePositiveInteger(takeOptionValue(argument, arguments.end()), option_name);
        }
        else if (looksLikeOption(*argument))
        {
            throw CommandLineError("Unknown option '" + std::string(*argument) + "'");
        }
        else if (image_path.has_value())
        {
            throw CommandLineError("Unexpected extra argument '" + std::string(*argument) + "'");
        }
        else
        {
            image_path = std::string(*argument);
        }
    }

    if (!image_path.has_value())
        throw CommandLineError("Missing image path");

    requireExistingFile(*image_path);
    options.image_path = *image_path;

    return options;
}

inline CommandLineOptions parseCommandLineOrExit(int argc, char *argv[])
{
    const auto program_name = command_line_detail::programNameFrom(argc, argv);

    try
    {
        const auto options = parseCommandLine(argc, argv);

        if (options.help_requested)
        {
            printHelp(program_name);
            std::exit(EXIT_SUCCESS);
        }

        return options;
    }
    catch (const CommandLineError &error)
    {
        std::cerr << "Error: " << error.what() << "\n\n";
        printHelp(program_name, std::cerr);
        std::exit(EXIT_FAILURE);
    }
}
