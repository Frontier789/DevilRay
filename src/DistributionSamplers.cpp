#include "tracing/DistributionSamplers.hpp"

#include "Image.hpp"
#include "Utils.hpp"
#include "device/Array.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <queue>
#include <span>
#include <utility>
#include <vector>

namespace
{
    constexpr int UNINITIALIZED = -1;

    void ensurePALarger(AliasEntry &entry)
    {
        if (entry.p_A < 0.5f) {
            entry.p_A = 1 - entry.p_A;
            std::swap(entry.A, entry.B);
            std::swap(entry.pdf_A, entry.pdf_B);
        }
    }

    void buildAliasTable(const std::vector<float> &normalized_importances, AliasEntry *entries)
    {
        const auto n = static_cast<int>(normalized_importances.size());
        const auto &importances = normalized_importances;
        if (importances.empty()) return;

        auto entriesPtr = entries;

        const auto overfullPrio  = [](const AliasEntry &a, const AliasEntry &b) {return a.p_A < b.p_A;};
        const auto underfullPrio = [](const AliasEntry &a, const AliasEntry &b) {return a.p_A < b.p_A;};

        std::priority_queue<AliasEntry, std::vector<AliasEntry>, decltype(overfullPrio)>  overfull(overfullPrio);
        std::priority_queue<AliasEntry, std::vector<AliasEntry>, decltype(underfullPrio)> underfull(underfullPrio);



        for (int i=0;i<importances.size();++i)
        {
            const float p = importances[i];

            const auto e = AliasEntry{
                .p_A = n*p,
                .pdf_A = 0,
                .pdf_B = 0,
                .A = i,
                .B = UNINITIALIZED,
            };

            if (e.p_A > 1) {
                overfull.push(e);
            }
            else if (e.p_A < 1) {
                underfull.push(e);
            }
            else {
                *entriesPtr++ = e;
            }
        }

        while (!underfull.empty() || !overfull.empty())
        {
            if (underfull.empty() || overfull.empty()) break;

            auto under = underfull.top();
            auto over = overfull.top();
            underfull.pop();
            overfull.pop();

            under.B = over.A;
            over.p_A = over.p_A + under.p_A - 1;

            ensurePALarger(under);
            *entriesPtr++ = std::move(under);

            if (over.p_A > 1) {
                overfull.push(over);
            }
            else if (over.p_A < 1) {
                underfull.push(over);
            }
            else {
                ensurePALarger(over);
                *entriesPtr++ = std::move(over);
            }
        }

        while (!underfull.empty()) {
            auto e = underfull.top();
            underfull.pop();

            e.p_A = 1;

            *entriesPtr++ = std::move(e);
        }

        while (!overfull.empty()) {
            auto e = overfull.top();
            overfull.pop();

            e.p_A = 1;

            *entriesPtr++ = std::move(e);
        }

        std::vector<float> pdfs(n);
        for (size_t i = 0; i < n; ++i)
            pdfs[i] = importances[i];

        for (size_t i = 0; i < n; ++i) {
            entries[i].pdf_A = pdfs[entries[i].A];
            if (entries[i].B >= 0 && entries[i].B < n)
                entries[i].pdf_B = pdfs[entries[i].B];
            else
                entries[i].pdf_B = 0.0f;
        }
    }

    struct Importances
    {
        std::vector<float> values;
        float total;
    };

    Importances normalizeImportances(std::span<const float> importances)
    {
        std::vector<float> filtered_importances;
        filtered_importances.reserve(importances.size());
        for (float f : importances) {
            if (f < 0.f) f = 0.f;
            if (std::isnan(f)) f = 0.f;
            if (std::isinf(f)) f = 0.f;

            filtered_importances.push_back(f);
        }

        const float total_importance = std::accumulate(filtered_importances.begin(), filtered_importances.end(), 0.f);

        if (total_importance > 0.0f) {
            for (float &f : filtered_importances) {
                f /= total_importance;
            }
        }

        return Importances{
            .values = std::move(filtered_importances),
            .total = total_importance
        };
    }
}

AliasTable AliasTable::make_uniform(int entry_count)
{
    if (entry_count <= 0) return make_empty();

    auto table = AliasTable{
        .entries = DeviceArray<AliasEntry>(entry_count, AliasEntry{}),
    };

    const float uniform_pdf = 1.0f / entry_count;
    AliasEntry *entries = table.entries.hostPtr();

    for (int i = 0; i < entry_count; ++i)
    {
        entries[i] = AliasEntry{
            .p_A = 1,
            .pdf_A = uniform_pdf,
            .pdf_B = 0,
            .A = i,
            .B = UNINITIALIZED,
        };
    }

    return table;
}

AliasTable generateAliasTable(std::span<const float> importances)
{
    const auto n = importances.size();

    const auto normalized_importances = normalizeImportances(importances);
    if (normalized_importances.total <= 0.0f) {
        return AliasTable::make_empty();
    }

    auto table = AliasTable{
        .entries = DeviceArray<AliasEntry>(n, AliasEntry{}),
    };

    buildAliasTable(normalized_importances.values, table.entries.hostPtr());

    return table;
}

AliasImageTable generateAliasTable(const ImageView<const float> &image)
{
    const auto s = image.size;

    auto table = AliasImageTable{
        .pixel_entries = DeviceArray<AliasEntry>(s.area(), AliasEntry{}),
        .row_entries = DeviceArray<AliasEntry>(s.height, AliasEntry{}),
        .image_size = s,
    };

    std::vector<float> row_sums(s.height, 0);

    parallel_for(s.height, [&](int i)
    {
        const auto w = static_cast<size_t>(s.width);
        const auto importances = std::span{image.pixels + w*i, w};

        const auto normalized_importances = normalizeImportances(importances);
        if (normalized_importances.total <= 0.0f) {
            row_sums[i] = 0.0f;
            return;
        }

        buildAliasTable(normalized_importances.values, table.pixel_entries.hostPtr() + w*i);
        row_sums[i] = normalized_importances.total;
    });

    const auto h = s.height;
    const auto importances = std::span{row_sums};

    const auto normalized_importances = normalizeImportances(importances);
    if (normalized_importances.total <= 0.0f) {
        return AliasImageTable::make_empty();
    }

    buildAliasTable(normalized_importances.values, table.row_entries.hostPtr());

    return table;
}
