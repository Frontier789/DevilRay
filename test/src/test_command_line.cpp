// AI-generated tests (Claude), reviewed by hand before committing.

#include "CommandLine.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
    constexpr const char *existing_image = "sample_image/lady.jpg";

    CommandLineOptions parse(std::vector<std::string> arguments_after_program_name)
    {
        std::vector<std::string> arguments{"sample_image"};
        arguments.insert(arguments.end(), arguments_after_program_name.begin(), arguments_after_program_name.end());

        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());

        return parseCommandLine(static_cast<int>(argv.size()), argv.data());
    }

    std::string errorMessageOfParsing(std::vector<std::string> arguments_after_program_name)
    {
        try
        {
            parse(std::move(arguments_after_program_name));
        }
        catch (const CommandLineError &error)
        {
            return error.what();
        }

        return "<parsed without error>";
    }

    bool contains(const std::string &text, const std::string &part)
    {
        return text.find(part) != std::string::npos;
    }
}

// --- Accepted command lines ---

TEST(CommandLineTest, OnlyTheImagePathIsRequiredAndEverythingElseKeepsItsDefault)
{
    const auto defaults = CommandLineOptions{};

    const auto options = parse({existing_image});

    EXPECT_EQ(options.image_path, existing_image);
    EXPECT_EQ(options.output_path, defaults.output_path);
    EXPECT_EQ(options.point_count, defaults.point_count);
    EXPECT_FALSE(options.plot_points);
    EXPECT_FALSE(options.sample_details);
    EXPECT_FALSE(options.help_requested);
}

TEST(CommandLineTest, FlagsHaveShortAndLongSpellings)
{
    for (const auto &plot_flag : {"-p", "--plot-points"})
        EXPECT_TRUE(parse({plot_flag, existing_image}).plot_points) << plot_flag;

    for (const auto &details_flag : {"-d", "--details"})
        EXPECT_TRUE(parse({details_flag, existing_image}).sample_details) << details_flag;
}

TEST(CommandLineTest, PointCountHasShortAndLongSpellings)
{
    EXPECT_EQ(parse({"-n", "42", existing_image}).point_count, 42);
    EXPECT_EQ(parse({"--point-count", "7", existing_image}).point_count, 7);
}

TEST(CommandLineTest, OutputPathHasShortAndLongSpellings)
{
    EXPECT_EQ(parse({"-o", "short.png", existing_image}).output_path, "short.png");
    EXPECT_EQ(parse({"--output", "long.png", existing_image}).output_path, "long.png");
}

TEST(CommandLineTest, OptionsMayFollowTheImagePath)
{
    const auto options = parse({existing_image, "-n", "9", "-o", "out.png", "-p", "-d"});

    EXPECT_EQ(options.image_path, existing_image);
    EXPECT_EQ(options.point_count, 9);
    EXPECT_EQ(options.output_path, "out.png");
    EXPECT_TRUE(options.plot_points);
    EXPECT_TRUE(options.sample_details);
}

TEST(CommandLineTest, HelpIsHonoredEvenWithoutAnImagePath)
{
    EXPECT_TRUE(parse({"-h"}).help_requested);
    EXPECT_TRUE(parse({"--help"}).help_requested);
    EXPECT_TRUE(parse({"-p", "-h"}).help_requested);
}

// --- Rejected command lines ---

TEST(CommandLineTest, PointCountMustBeAPositiveWholeNumber)
{
    for (const auto &bad_count : {"0", "-3", "abc", "12abc", "1.5", ""})
    {
        const auto message = errorMessageOfParsing({"-n", bad_count, existing_image});

        EXPECT_TRUE(contains(message, "-n expects a positive integer")) << "'" << bad_count << "': " << message;
    }
}

TEST(CommandLineTest, EveryOptionTakingAValueRejectsBeingTheLastArgument)
{
    for (const auto &option : {"-n", "--point-count", "-o", "--output"})
    {
        const auto message = errorMessageOfParsing({existing_image, option});

        EXPECT_TRUE(contains(message, std::string(option) + " expects a value")) << message;
    }
}

TEST(CommandLineTest, UnknownOptionIsRejectedAndNamed)
{
    const auto message = errorMessageOfParsing({"--bogus", existing_image});

    EXPECT_TRUE(contains(message, "Unknown option '--bogus'")) << message;
}

TEST(CommandLineTest, MissingImagePathIsRejected)
{
    EXPECT_TRUE(contains(errorMessageOfParsing({}), "Missing image path"));
    EXPECT_TRUE(contains(errorMessageOfParsing({"-p"}), "Missing image path"));
}

TEST(CommandLineTest, SecondPositionalArgumentIsRejectedAndNamed)
{
    const auto message = errorMessageOfParsing({existing_image, "extra.jpg"});

    EXPECT_TRUE(contains(message, "Unexpected extra argument 'extra.jpg'")) << message;
}

TEST(CommandLineTest, ImageThatDoesNotExistIsRejectedAndNamed)
{
    const auto message = errorMessageOfParsing({"sample_image/no_such_image.jpg"});

    EXPECT_TRUE(contains(message, "Image file not found: 'sample_image/no_such_image.jpg'")) << message;
}

// --- Help text ---

TEST(CommandLineTest, HelpTextListsEveryOptionAndTheDefaultsTheParserUses)
{
    const auto defaults = CommandLineOptions{};
    std::ostringstream help;

    printHelp("sample_image", help);

    const auto option_lines = {
        "  -n, --point-count",
        "  -p, --plot-points",
        "  -d, --details",
        "  -o, --output",
        "  -h, --help",
    };
    for (const auto &option_line : option_lines)
        EXPECT_TRUE(contains(help.str(), option_line)) << option_line;

    EXPECT_TRUE(contains(help.str(), "default: " + std::to_string(defaults.point_count)));
    EXPECT_TRUE(contains(help.str(), "default: " + defaults.output_path));
}
