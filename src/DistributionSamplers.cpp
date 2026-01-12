#include "tracing/DistributionSamplers.hpp"

#include <numeric>
#include <queue>

namespace
{
    constexpr int UNINITIALIZED = -1;

    AliasTable generateUniformAliasTable(int n)
    {
        DeviceArray<AliasEntry> table(n, AliasEntry{});

        for (int i=0; i<n; ++i)
        {
            table.hostPtr()[i] = AliasEntry{
                .p_A = 1,
                .A = i,
                .B = UNINITIALIZED 
            };
        }

        return AliasTable{ .entries = std::move(table) };
    }
}

AliasTable generateAliasTable(std::span<const float> importances)
{
    const auto n = importances.size();
    
    std::vector<float> filteredImportances;
    filteredImportances.reserve(n);
    for (const auto &f : importances) filteredImportances.push_back(std::max(0.f, f));

    const float totalImportance = std::accumulate(filteredImportances.begin(), filteredImportances.end(), 0.f);

    AliasTable table{
        .entries = DeviceArray<AliasEntry>(n, AliasEntry{})
    };
    table.entries.ensureDeviceAllocation();

    if (n == 0) return table;
    if (totalImportance < 1e-5f) return generateUniformAliasTable(n);

    auto entriesPtr = table.entries.hostPtr();
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

        *entriesPtr++ = std::move(under);

        if (over.p_A > 1) {
            overfull.push(over);
        }
        else if (over.p_A < 1) {
            underfull.push(over);
        }
        else {
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
    
    return table;
}
