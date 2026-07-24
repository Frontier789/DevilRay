# AI-generated (Claude), reviewed by hand before committing.
#
# GitVersion.cmake — two modes in one file.
#
#  * include()d from a CMakeLists it defines target_git_version(target source),
#    which wires up a build-time step that regenerates <build>/generated/
#    GitVersion.hpp with the current commit hash and branch, makes `source`
#    depend on it, and puts the header on `target`'s private include path.
#
#  * Run via `cmake -P` (which the step above does internally) it performs that
#    regeneration: it rewrites the header only when the commit/branch change,
#    leaves the header absent when git is unavailable (the consumer guards it
#    with __has_include, so those builds stay clean), and touches a stamp file
#    only on change so the metadata TU recompiles at most once per commit.
#
# Keeping both here means the top-level CMakeLists only needs a single call.

if(CMAKE_SCRIPT_MODE_FILE)
    # ======================= build-time script mode =======================
    find_package(Git QUIET)

    set(GIT_COMMIT "")
    set(GIT_BRANCH "")
    set(GIT_ORIGIN "")

    if(Git_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
            WORKING_DIRECTORY "${SOURCE_DIR}"
            OUTPUT_VARIABLE GIT_COMMIT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE commit_status)

        execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY "${SOURCE_DIR}"
            OUTPUT_VARIABLE GIT_BRANCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE branch_status)

        execute_process(
            COMMAND "${GIT_EXECUTABLE}" config --get remote.origin.url
            WORKING_DIRECTORY "${SOURCE_DIR}"
            OUTPUT_VARIABLE GIT_ORIGIN
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE origin_status)

        if(NOT commit_status EQUAL 0)
            set(GIT_COMMIT "")
        endif()
        if(NOT branch_status EQUAL 0)
            set(GIT_BRANCH "")
        endif()
        if(NOT origin_status EQUAL 0)
            set(GIT_ORIGIN "")
        endif()
    endif()

    # If we couldn't fetch the essentials (no git, not a repo, command failed),
    # fail gracefully: report it, delete the header so __has_include drops the
    # git metadata, keep the stamp valid, and stop. Never error out the build.
    if(GIT_COMMIT STREQUAL "")
        message(STATUS "GitVersion: failed to fetch git info; git metadata omitted")
        set(changed OFF)
        if(EXISTS "${GIT_HEADER}")
            file(REMOVE "${GIT_HEADER}")
            set(changed ON)
        endif()
        if(changed OR NOT EXISTS "${GIT_STAMP}")
            file(WRITE "${GIT_STAMP}" "\n")
        endif()
        return()
    endif()

    # Normalize the remote to a clickable https:// link. git only reports the
    # remote as configured (often SSH), so convert the SSH forms ourselves.
    if(NOT GIT_ORIGIN STREQUAL "")
        if(NOT GIT_ORIGIN MATCHES "://")
            # scp-like syntax: [user@]host:path -> https://host/path
            if(GIT_ORIGIN MATCHES "^[^@]+@([^:]+):(.+)$")
                set(GIT_ORIGIN "https://${CMAKE_MATCH_1}/${CMAKE_MATCH_2}")
            endif()
        elseif(GIT_ORIGIN MATCHES "^ssh://")
            # ssh://[user@]host/path -> https://host/path
            string(REGEX REPLACE "^ssh://([^@]+@)?" "https://" GIT_ORIGIN "${GIT_ORIGIN}")
        endif()

        # Drop the trailing .git so the link opens the project page directly.
        string(REGEX REPLACE "\\.git$" "" GIT_ORIGIN "${GIT_ORIGIN}")
    endif()

    set(NEW_CONTENT
"#pragma once
// Auto-generated at build time by cmake/GitVersion.cmake. Do not edit.
#define DEVILRAY_GIT_COMMIT \"${GIT_COMMIT}\"
#define DEVILRAY_GIT_BRANCH \"${GIT_BRANCH}\"
")
    # A repository can have a commit but no remote, so origin is optional.
    if(NOT GIT_ORIGIN STREQUAL "")
        string(APPEND NEW_CONTENT "#define DEVILRAY_GIT_ORIGIN \"${GIT_ORIGIN}\"\n")
    endif()

    # Only rewrite the header (and touch the stamp) when the content changes, so
    # the metadata TU recompiles at most once per commit.
    set(changed OFF)
    set(OLD_CONTENT "")
    if(EXISTS "${GIT_HEADER}")
        file(READ "${GIT_HEADER}" OLD_CONTENT)
    endif()
    if(NOT OLD_CONTENT STREQUAL NEW_CONTENT)
        file(WRITE "${GIT_HEADER}" "${NEW_CONTENT}")
        set(changed ON)
        message(STATUS "GitVersion: ${GIT_BRANCH} @ ${GIT_COMMIT}")
    endif()

    if(changed OR NOT EXISTS "${GIT_STAMP}")
        file(WRITE "${GIT_STAMP}" "${GIT_COMMIT}\n")
    endif()

    return()
endif()

# ====================== configure-time module mode ======================

# Wires a build-time git-version header into `target`, consumed by `source`.
function(target_git_version target source)
    set(gen_dir "${CMAKE_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${gen_dir}")
    set(header "${gen_dir}/GitVersion.hpp")
    set(stamp "${gen_dir}/git_version.stamp")

    # The never-created phony output re-runs this on every build, so new commits
    # are picked up without reconfiguring. The header is a BYPRODUCT so the
    # compiler-discovered dependency on it resolves; the build itself depends on
    # the always-present stamp (the header may be absent without git). The stamp
    # changes only when the git info does, so with Ninja's restat only `source`
    # recompiles, and only then.
    add_custom_command(
        OUTPUT "${stamp}" "${gen_dir}/.git_version_phony"
        BYPRODUCTS "${header}"
        COMMAND ${CMAKE_COMMAND}
                -DGIT_HEADER=${header}
                -DGIT_STAMP=${stamp}
                -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GitVersion.cmake
        COMMENT "Refreshing git version header"
        VERBATIM)

    set_source_files_properties("${source}" PROPERTIES OBJECT_DEPENDS "${stamp}")
    target_include_directories(${target} PRIVATE "${gen_dir}")
endfunction()
