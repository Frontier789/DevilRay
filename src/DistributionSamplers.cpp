#include "tracing/DistributionSamplers.hpp"

#include <numeric>
#include <queue>

namespace
{
    constexpr int UNINITIALIZED = -1;

    void buildUniformAliasTable(AliasEntry *entries, int n)
    {
        const float uniformPdf = (n > 0) ? 1.0f / n : 0.0f;

        for (int i=0; i<n; ++i)
        {
            entries[i] = AliasEntry{
                .p_A = 1,
                .pdf_A = uniformPdf,
                .pdf_B = 0,
                .A = i,
                .B = UNINITIALIZED
            };
        }
    }

    void ensurePALarger(AliasEntry &entry)
    {
        if (entry.p_A < 0.5f) {
            entry.p_A = 1 - entry.p_A;
            std::swap(entry.A, entry.B);
            std::swap(entry.pdf_A, entry.pdf_B);
        }
    }

    void buildAliasTable(const float *importances, AliasEntry *entries, int n)
    {
        std::vector<float> filteredImportances;
        filteredImportances.reserve(n);
        for (const auto &f : std::span{importances, static_cast<size_t>(n)}) {
            filteredImportances.push_back(std::max(0.f, f));
        }

        const float totalImportance = std::accumulate(filteredImportances.begin(), filteredImportances.end(), 0.f);

        if (n == 0) return;
        if (totalImportance < 1e-5f) return buildUniformAliasTable(entries, n);

        auto entriesPtr = entries;
        int tableIndex = 0;

        const auto overfullPrio  = [](const AliasEntry &a, const AliasEntry &b) {return a.p_A < b.p_A;};
        const auto underfullPrio = [](const AliasEntry &a, const AliasEntry &b) {return a.p_A < b.p_A;};

        std::priority_queue<AliasEntry, std::vector<AliasEntry>, decltype(overfullPrio)>  overfull(overfullPrio);
        std::priority_queue<AliasEntry, std::vector<AliasEntry>, decltype(underfullPrio)> underfull(underfullPrio);



        for (int i=0;i<n;++i)
        {
            const float p = filteredImportances[i] / totalImportance;

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
            pdfs[i] = filteredImportances[i] / totalImportance;

        for (size_t i = 0; i < n; ++i) {
            entries[i].pdf_A = pdfs[entries[i].A];
            if (entries[i].B >= 0 && entries[i].B < n)
                entries[i].pdf_B = pdfs[entries[i].B];
            else
                entries[i].pdf_B = 0.0f;
        }
    }
}

AliasTable generateAliasTable(std::span<const float> importances)
{
    const auto n = importances.size();

    auto table = AliasTable{
        .entries = DeviceArray<AliasEntry>(n, AliasEntry{}),
    };

    buildAliasTable(importances.data(), table.entries.hostPtr(), n);

    return table;
}

AliasImageTable generateAliasTable(const ImageView1f &image)
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
        const auto w = s.width;
        buildAliasTable(image.pixels + w*i, table.pixel_entries.hostPtr() + w*i, w);

        row_sums[i] = std::accumulate(image.pixels + w*i, image.pixels + w*(i+1), 0.0f);
    });

    const auto h = s.height;
    buildAliasTable(row_sums.data(), table.row_entries.hostPtr(), h);

    return table;
}
