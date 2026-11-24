// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace VideoCore {
class BufferCache;
}

namespace Vulkan {

class Instance;

/// How a ZPASS predicate (SET_PREDICATION Zpass) resolves where a draw is recorded.
struct ZpassPredicate {
    enum class Kind : u8 {
        Unknown,  ///< no results to go by: draw
        Visible,  ///< known on the CPU
        Occluded, ///< known on the CPU
        Gpu,      ///< conditional rendering on PredicateBuffer() at `offset`
    };
    Kind kind{Kind::Unknown};
    u32 offset{};
};

/// bbport: the DB pixel counters of ZPASS_DONE / PIXEL_PIPE_STAT_DUMP as Vulkan occlusion
/// queries. Draws run inside queries (begun before a draw, ended at the next dump and at every
/// boundary a query may not span: Scheduler::CarriedScope). A dump notes the queries recorded
/// before it; once the GPU has run its submission, a resolver thread sums their results and
/// writes the counters into guest memory, valid bit set, as the DBs do. A predicate over the
/// range between two dumps is evaluated on the CPU when both are written, else by a compute pass
/// (occlusion_predicate.comp) into a buffer for VK_EXT_conditional_rendering.
///
/// Recording side only (the thread recording draws: Scheduler::Record rules), except where noted.
/// Counting starts with the first dump: before it nobody reads the counters.
/// BB_OCCLUSION_QUERIES=0: no queries, counters advanced by a constant as before.
class OcclusionCounter final : public Scheduler::CarriedScope {
public:
    explicit OcclusionCounter(const Instance& instance, Scheduler& scheduler,
                              VideoCore::BufferCache& buffer_cache, u32 num_dbs);
    ~OcclusionCounter();

    OcclusionCounter(const OcclusionCounter&) = delete;
    OcclusionCounter& operator=(const OcclusionCounter&) = delete;

    /// A counter dump at `address`: `num_dbs` counters `stride` bytes apart.
    void Dump(VAddr address, u32 num_dbs, u32 stride);
    /// Inside the render pass, before a draw is recorded. `db_count_control`: DB_COUNT_CONTROL.
    void BeforeDraw(u32 db_count_control);
    /// Scheduler::CarriedScope: a query ends at every boundary; the next draw begins another.
    void Suspend(bool inside_render_pass) override;
    void Resume(bool inside_render_pass) override;
    /// Submit callback: the dumps recorded so far are in the submission signalling `tick`.
    void OnSubmit(u64 tick);
    /// Any thread: dumps recorded but not submitted yet (the guest may wait for their results).
    [[nodiscard]] bool HasUnsubmittedDumps();

    /// Outside a draw's render pass (it may record a compute pass).
    ZpassPredicate EvaluateZpass(VAddr address, bool draw_if_visible);

    [[nodiscard]] vk::Buffer PredicateBuffer() const noexcept {
        return predicate_buffer.buffer;
    }

private:
    struct Record {
        VAddr address;
        u64 id;
        u64 query_end;
        u64 tick;
        u32 num_dbs;
        u32 stride;
    };

    struct DumpSlot {
        VAddr address{};
        u64 id{};
        u64 query_end{};
        u64 uncounted{};
        u64 job_seq{};
        u64 total{};
        u64 lost{};
        bool resolved{};
    };

    struct Job {
        u64 seq{~0ULL};
        u64 first{};
        u32 count{};
        bool computed{};
    };

    void End();
    void WriteFake(VAddr address, u32 num_dbs, u32 stride);
    /// Guest memory through the backing view (no faults on GPU-written pages).
    static bool ReadGuest(VAddr address, u64& value);
    void WriteGuest(VAddr address, u64 value);
    DumpSlot& SlotFor(VAddr address);

    void ResolverThread(std::stop_token stoken);
    bool WaitTick(u64 tick, std::stop_token stoken);
    bool Resolve(const Record& record, std::stop_token stoken);

    void CreatePredicatePass();
    void ComputePredicates();

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::BufferCache& buffer_cache;
    const u32 num_dbs;
    vk::QueryControlFlags query_flags;
    vk::UniqueQueryPool pool;

    bool armed{};  ///< a dump was seen: draws are counted
    bool active{}; ///< a query is open in the command stream
    u64 query_write{};
    std::atomic<u64> query_read{};
    u64 next_dump_id{1};
    u64 uncounted{};
    u64 fake_counter{};

    std::mutex mutex;
    std::condition_variable_any records_cv;
    bool is_resolving{false};
    std::vector<Record> records;
    std::vector<Record> pending_records;
    std::atomic<u32> num_pending_records{0};
    std::unordered_map<VAddr, DumpSlot> dump_slots;
    u64 hold_begin{~0ULL};
    u64 hold_end{};
    u64 hold_tick{};

    u64 total{};
    u64 lost{};
    std::vector<u64> results;

    bool predicates_enabled{};
    VideoCore::UniqueBuffer predicate_buffer;
    vk::UniqueDescriptorSetLayout predicate_desc_layout;
    vk::UniquePipelineLayout predicate_pipeline_layout;
    vk::UniquePipeline predicate_pipeline;
    std::vector<Job> jobs;
    u64 next_job_seq{};
    u64 computed_seq{};

    std::jthread resolver;
};

} // namespace Vulkan
