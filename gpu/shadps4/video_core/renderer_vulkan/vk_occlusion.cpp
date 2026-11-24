// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>

#include "common/logging/log.h"
#include "common/thread.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_occlusion.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include "video_core/host_shaders/occlusion_predicate_comp.h"

extern "C" void runtime_memory_read_backing(std::uintptr_t, void*, std::uint64_t);

namespace Vulkan {

namespace {

constexpr u32 NumQueries = 16384;
constexpr u64 WaitSliceNs = 5'000'000;
constexpr u64 ValidBit = 1ULL << 63;
constexpr u32 DumpStride = 16;
constexpr u32 ZpassIncrementDisable = 1; ///< DB_COUNT_CONTROL.ZPASS_INCREMENT_DISABLE
constexpr u64 FakeCounterStep = 0x2FFFFFFULL;

constexpr u32 NumJobs = 4096;
constexpr u32 ResultsOffset = 0;
constexpr u32 JobsOffset = ResultsOffset + NumQueries * 2 * sizeof(u32);
constexpr u32 PredsOffset = JobsOffset + NumJobs * 2 * sizeof(u32);
constexpr u32 PredicateBufferSize = PredsOffset + NumJobs * 2 * sizeof(u32);
constexpr u32 PredicateGroupSize = 64;

struct PredicatePushConstants {
    u32 job_count;
    u32 out_start;
    u32 num_queries;
    u32 num_preds;
    u32 jobs_base;
    u32 preds_base;
};

bool QueriesWanted() {
    static const bool wanted = [] {
        const char* env = std::getenv("BB_OCCLUSION_QUERIES");
        return !(env && env[0] == '0');
    }();
    return wanted;
}

} // Anonymous namespace

OcclusionCounter::OcclusionCounter(const Instance& instance_, Scheduler& scheduler_,
                                   VideoCore::BufferCache& buffer_cache_, u32 num_dbs_)
    : instance{instance_}, scheduler{scheduler_}, buffer_cache{buffer_cache_}, num_dbs{num_dbs_},
      predicate_buffer{instance_.GetDevice(), instance_.GetAllocator()} {
    if (!QueriesWanted()) {
        std::printf("Occlusion: queries off (BB_OCCLUSION_QUERIES=0), counters advance by a "
                    "constant\n");
        return;
    }
    if (!instance.IsHostQueryResetSupported()) {
        LOG_WARNING(Render_Vulkan, "Occlusion queries need hostQueryReset; counters are faked");
        return;
    }
    const vk::Device device = instance.GetDevice();
    auto [result, query_pool] = device.createQueryPoolUnique({
        .queryType = vk::QueryType::eOcclusion,
        .queryCount = NumQueries,
    });
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Creating the occlusion query pool failed: {}",
                    vk::to_string(result));
        return;
    }
    pool = std::move(query_pool);
    query_flags = {};
    device.resetQueryPool(*pool, 0, NumQueries);
    results.resize(NumQueries * 2);
    if (instance.IsConditionalRenderingSupported()) {
        CreatePredicatePass();
    }
    resolver = std::jthread(std::bind_front(&OcclusionCounter::ResolverThread, this));
    scheduler.SetCarriedScope(this);
}

OcclusionCounter::~OcclusionCounter() {
    if (pool) {
        scheduler.SetCarriedScope(nullptr);
    }
    if (resolver.joinable()) {
        resolver.request_stop();
        records_cv.notify_all();
        resolver.join();
    }
}

void OcclusionCounter::CreatePredicatePass() {
    const vk::Device device = instance.GetDevice();
    vk::BufferCreateInfo buffer_ci = {
        .size = PredicateBufferSize,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                 vk::BufferUsageFlagBits::eConditionalRenderingEXT,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    predicate_buffer.Create(buffer_ci, VideoCore::MemoryType::DeviceLocal, nullptr);
    SetObjectName(device, predicate_buffer.buffer, "Occlusion Predicates");

    const vk::DescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
    };
    predicate_desc_layout = Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = 1,
        .pBindings = &binding,
    }));
    const vk::PushConstantRange push_range = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(PredicatePushConstants),
    };
    const vk::DescriptorSetLayout set_layout = *predicate_desc_layout;
    predicate_pipeline_layout = Check(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    }));
    const vk::ShaderModule module = CompileSPV(OCCLUSION_PREDICATE_COMP, device);
    const vk::ComputePipelineCreateInfo pipeline_ci = {
        .stage =
            {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = module,
                .pName = "main",
            },
        .layout = *predicate_pipeline_layout,
    };
    predicate_pipeline = Check(device.createComputePipelineUnique({}, pipeline_ci));
    device.destroyShaderModule(module);

    jobs.resize(NumJobs);
    predicates_enabled = true;
}

OcclusionCounter::DumpSlot& OcclusionCounter::SlotFor(VAddr address) {
    auto& slot = dump_slots[address];
    slot.address = address;
    return slot;
}

bool OcclusionCounter::ReadGuest(VAddr address, u64& value) {
    auto* memory = Core::Memory::Instance();
    if (address == 0 || (address & 7) != 0 || !memory ||
        memory->ClampRangeSize(address, sizeof(u64)) != sizeof(u64)) {
        return false;
    }
    runtime_memory_read_backing(address, &value, sizeof(u64));
    return true;
}

void OcclusionCounter::WriteGuest(VAddr address, u64 value) {
    auto* memory = Core::Memory::Instance();
    if (address == 0 || (address & 7) != 0 || !memory ||
        memory->ClampRangeSize(address, sizeof(u64)) != sizeof(u64)) {
        return;
    }
    auto* ptr = reinterpret_cast<u64*>(address);
    if (!memory->TryWriteBacking(ptr, &value, sizeof(value))) {
        __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
    }
    // A VRAM copy of the results gets them too (any thread).
    buffer_cache.NoteLateCommandWrite(address, &value, sizeof(value));
}

void OcclusionCounter::WriteFake(VAddr address, u32 dbs, u32 stride) {
    for (u32 db = 0; db < dbs; ++db) {
        WriteGuest(address + u64(db) * stride, fake_counter | ValidBit);
    }
    fake_counter += FakeCounterStep;
}

void OcclusionCounter::Dump(VAddr address, u32 dbs, u32 stride) {
    dbs = dbs ? dbs : num_dbs;
    stride = stride ? stride : DumpStride;
    if (!pool) {
        WriteFake(address, dbs, stride);
        return;
    }
    armed = true;
    End();
    std::scoped_lock lk{mutex};
    const u64 id = next_dump_id++;
    pending_records.push_back({
        .address = address,
        .id = id,
        .query_end = query_write,
        .tick = 0,
        .num_dbs = dbs,
        .stride = stride,
    });
    num_pending_records.store(u32(pending_records.size()), std::memory_order_release);
    // The begin dump of a ZPASS predicate is 8 bytes before the end dump (one DB's pair).
    u64 job_seq = ~0ULL;
    auto it = dump_slots.find(address - 8);
    if (it == dump_slots.end() && stride > 8) {
        it = dump_slots.find(address - stride / 2);
    }
    if (it == dump_slots.end()) {
        it = dump_slots.find(address - 16);
    }
    if (it != dump_slots.end() && it->second.query_end < query_write && predicates_enabled) {
        const DumpSlot& begin = it->second;
        job_seq = next_job_seq++;
        jobs[job_seq % NumJobs] = {
            .seq = job_seq,
            .first = begin.query_end,
            .count = static_cast<u32>(query_write - begin.query_end),
        };
    }
    SlotFor(address) = {
        .address = address,
        .id = id,
        .query_end = query_write,
        .uncounted = uncounted,
        .job_seq = job_seq,
    };

    if (dump_slots.size() > 8192) {
        std::erase_if(dump_slots, [&](const auto& pair) {
            return pair.second.resolved && (id - pair.second.id > 4096);
        });
    }
}

void OcclusionCounter::OnSubmit(u64 tick) {
    {
        std::scoped_lock lk{mutex};
        if (pending_records.empty()) {
            return;
        }
        for (auto& record : pending_records) {
            record.tick = tick;
            records.push_back(record);
        }
        pending_records.clear();
        num_pending_records.store(0, std::memory_order_release);
    }
    records_cv.notify_one();
}

bool OcclusionCounter::HasUnsubmittedDumps() {
    return num_pending_records.load(std::memory_order_acquire) != 0;
}

void OcclusionCounter::BeforeDraw(u32 db_count_control) {
    if (!pool || !armed) {
        return;
    }
    if (db_count_control & ZpassIncrementDisable) {
        End();
        return;
    }
    if (active) {
        return;
    }
    if (query_write - query_read.load(std::memory_order_acquire) >= NumQueries) {
        LOG_TRACE(Render_Vulkan, "Occlusion query pool exhausted ({}/{}), skipping query",
                  NumQueries, NumQueries);
        ++uncounted;
        return;
    }
    const vk::QueryPool handle = *pool;
    const u32 slot = static_cast<u32>(query_write % NumQueries);
    const vk::QueryControlFlags flags = query_flags;
    scheduler.Record([handle, slot, flags](vk::CommandBuffer cmdbuf) {
        cmdbuf.beginQuery(handle, slot, flags);
    });
    active = true;
}

void OcclusionCounter::Suspend(bool) {
    End();
}

void OcclusionCounter::Resume(bool) {
    // The next draw begins a new query (queries are begun inside render passes only).
}

void OcclusionCounter::End() {
    if (!active) {
        return;
    }
    active = false; // before Record: a boundary inside it must not end the query twice
    const vk::QueryPool handle = *pool;
    const u32 slot = static_cast<u32>(query_write % NumQueries);
    scheduler.Record([handle, slot](vk::CommandBuffer cmdbuf) { cmdbuf.endQuery(handle, slot); });
    ++query_write;
}

ZpassPredicate OcclusionCounter::EvaluateZpass(VAddr address, bool draw_if_visible) {
    if (!pool) {
        return {};
    }
    // Both dumps written already (the guest may also have copied the results there).
    u64 begin_val = 0, end_val = 0;
    const bool have_begin = ReadGuest(address, begin_val);
    bool have_end = ReadGuest(address + 8, end_val);
    if (!have_end || !(end_val & ValidBit)) {
        have_end = ReadGuest(address + 16, end_val);
    }
    if (have_begin && have_end && (begin_val & ValidBit) && (end_val & ValidBit)) {
        const u64 passed = (end_val & ~ValidBit) - (begin_val & ~ValidBit);
        return {passed > 0 ? ZpassPredicate::Kind::Visible : ZpassPredicate::Kind::Occluded, 0};
    }
    u64 job_seq;
    {
        std::scoped_lock lk{mutex};
        auto it_begin = dump_slots.find(address);
        if (it_begin == dump_slots.end()) {
            LOG_TRACE(Render_Vulkan, "EvaluateZpass: no begin slot for addr={:#x}", address);
            return {};
        }
        auto it_end = dump_slots.find(address + 8);
        if (it_end == dump_slots.end()) {
            it_end = dump_slots.find(address + 16);
        }
        if (it_end == dump_slots.end()) {
            LOG_TRACE(Render_Vulkan, "EvaluateZpass: no end slot for addr={:#x}", address);
            return {};
        }
        const DumpSlot& begin = it_begin->second;
        const DumpSlot& end = it_end->second;
        if (end.id <= begin.id || end.uncounted != begin.uncounted) {
            LOG_TRACE(Render_Vulkan,
                      "EvaluateZpass: query slot mismatch or uncounted query for addr={:#x}",
                      address);
            return {};
        }
        if (end.resolved) {
            if (!begin.resolved || end.lost != begin.lost) {
                return {};
            }
            return {end.total != begin.total ? ZpassPredicate::Kind::Visible
                                             : ZpassPredicate::Kind::Occluded};
        }
        if (end.query_end == begin.query_end) {
            // No draw counted between the dumps: nothing passed.
            return {ZpassPredicate::Kind::Occluded};
        }
        job_seq = end.job_seq;
    }
    if (!predicates_enabled || job_seq == ~0ULL) {
        return {};
    }
    const Job& job = jobs[job_seq % NumJobs];
    if (job.seq != job_seq) {
        return {};
    }
    if (job_seq >= computed_seq) {
        ComputePredicates();
    }
    if (!job.computed) {
        return {};
    }
    const u32 word = static_cast<u32>(job_seq % NumJobs) * 2 + (draw_if_visible ? 0 : 1);
    return {ZpassPredicate::Kind::Gpu, PredsOffset + word * static_cast<u32>(sizeof(u32))};
}

void OcclusionCounter::ComputePredicates() {
    const u64 start = std::max(computed_seq, next_job_seq > NumJobs ? next_job_seq - NumJobs : 0);
    const u32 job_count = static_cast<u32>(next_job_seq - start);
    computed_seq = next_job_seq;
    if (job_count == 0) {
        return;
    }

    // Copies and dispatches run outside render passes; this also ends the open query.
    scheduler.EndRendering();
    End();

    u64 copy_first = ~0ULL;
    for (u64 seq = start; seq < next_job_seq; ++seq) {
        copy_first = std::min(copy_first, jobs[seq % NumJobs].first);
    }
    const u64 copy_end = query_write;
    {
        // The resolver must not reset the copied queries before the GPU has copied them.
        std::scoped_lock lk{mutex};
        copy_first = std::max(copy_first, query_read.load(std::memory_order_acquire));
        if (copy_first < copy_end) {
            hold_begin = std::min(hold_begin, copy_first);
            hold_end = std::max(hold_end, copy_end);
            hold_tick = scheduler.CurrentTick();
        }
    }

    std::vector<u32> job_upload;
    job_upload.reserve(job_count * 2);
    for (u64 seq = start; seq < next_job_seq; ++seq) {
        Job& job = jobs[seq % NumJobs];
        // Queries already read back and reset cannot be copied: such a job stays unknown.
        job.computed = job.first >= copy_first;
        job_upload.push_back(static_cast<u32>(job.first % NumQueries));
        job_upload.push_back(job.computed ? job.count : 0);
    }

    const vk::Buffer buffer = predicate_buffer.buffer;
    const vk::QueryPool query_pool = *pool;
    const vk::Pipeline pipeline = *predicate_pipeline;
    const vk::PipelineLayout layout = *predicate_pipeline_layout;
    const PredicatePushConstants push = {
        .job_count = job_count,
        .out_start = static_cast<u32>(start % NumJobs),
        .num_queries = NumQueries,
        .num_preds = NumJobs,
        .jobs_base = JobsOffset / sizeof(u32),
        .preds_base = PredsOffset / sizeof(u32),
    };
    scheduler.Record([buffer, query_pool, pipeline, layout, push, copy_first, copy_end,
                      job_upload = std::move(job_upload)](vk::CommandBuffer cmdbuf) {
        const vk::MemoryBarrier2 to_transfer = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eConditionalRenderingReadEXT |
                             vk::AccessFlagBits2::eShaderStorageRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        };
        cmdbuf.pipelineBarrier2({
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &to_transfer,
        });
        for (u64 query = copy_first; query < copy_end;) {
            const u32 slot = static_cast<u32>(query % NumQueries);
            const u32 count = static_cast<u32>(std::min<u64>(copy_end - query, NumQueries - slot));
            cmdbuf.copyQueryPoolResults(
                query_pool, slot, count, buffer, ResultsOffset + slot * 2 * sizeof(u32),
                2 * sizeof(u32),
                vk::QueryResultFlagBits::eWithAvailability | vk::QueryResultFlagBits::eWait);
            query += count;
        }
        cmdbuf.updateBuffer(buffer, JobsOffset, job_upload.size() * sizeof(u32),
                            job_upload.data());
        const vk::MemoryBarrier2 to_compute = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask =
                vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
        };
        cmdbuf.pipelineBarrier2({
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &to_compute,
        });

        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        const vk::DescriptorBufferInfo buffer_info = {
            .buffer = buffer,
            .offset = 0,
            .range = PredicateBufferSize,
        };
        const vk::WriteDescriptorSet write = {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &buffer_info,
        };
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, write);
        cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
        cmdbuf.dispatch((push.job_count + PredicateGroupSize - 1) / PredicateGroupSize, 1, 1);

        const vk::MemoryBarrier2 to_predicate = {
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eConditionalRenderingEXT,
            .dstAccessMask = vk::AccessFlagBits2::eConditionalRenderingReadEXT,
        };
        cmdbuf.pipelineBarrier2({
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &to_predicate,
        });
    });
}

void OcclusionCounter::ResolverThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:OcclusionResolver");
    std::vector<Record> batch;
    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{mutex};
            if (!records_cv.wait(lk, stoken, [this] { return !records.empty(); })) {
                return;
            }
            batch.swap(records);
            is_resolving = true;
        }
        for (const Record& record : batch) {
            if (!WaitTick(record.tick, stoken) || !Resolve(record, stoken)) {
                return;
            }
        }
        batch.clear();
        {
            std::unique_lock lk{mutex};
            is_resolving = false;
        }
    }
}

bool OcclusionCounter::WaitTick(u64 tick, std::stop_token stoken) {
    const vk::Device device = instance.GetDevice();
    const vk::Semaphore semaphore = scheduler.GetWorkSemaphore()->Handle();
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore,
        .pValues = &tick,
    };
    while (true) {
        const vk::Result result = device.waitSemaphores(wait_info, WaitSliceNs);
        if (result == vk::Result::eSuccess) {
            return true;
        }
        if (result != vk::Result::eTimeout) {
            LOG_ERROR(Render_Vulkan, "Occlusion resolver: waiting for tick {} failed: {}", tick,
                      vk::to_string(result));
            return false;
        }
        if (stoken.stop_requested()) {
            return false;
        }
    }
}

bool OcclusionCounter::Resolve(const Record& record, std::stop_token stoken) {
    const vk::Device device = instance.GetDevice();
    const u64 first = query_read.load(std::memory_order_relaxed);
    bool complete = true;
    for (u64 query = first; query < record.query_end;) {
        const u32 slot = query % NumQueries;
        const u32 count =
            static_cast<u32>(std::min<u64>(record.query_end - query, NumQueries - slot));
        const vk::Result result = device.getQueryPoolResults(
            *pool, slot, count, count * 2 * sizeof(u64), results.data(), 2 * sizeof(u64),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability |
                vk::QueryResultFlagBits::eWait);
        const bool read = result == vk::Result::eSuccess || result == vk::Result::eNotReady;
        for (u32 i = 0; i < count; ++i) {
            if (read && results[2 * i + 1] != 0) {
                total += results[2 * i];
            } else {
                complete = false;
                ++lost;
            }
        }
        query += count;
    }

    {
        std::unique_lock lk{mutex};
        // A predicate pass copies some of these queries: not reset before it has run.
        while (first < record.query_end && first < hold_end && record.query_end > hold_begin) {
            const u64 tick = hold_tick;
            lk.unlock();
            if (!WaitTick(tick, stoken)) {
                return false;
            }
            lk.lock();
            if (hold_tick == tick) {
                hold_begin = ~0ULL;
                hold_end = 0;
            }
        }
        for (u64 query = first; query < record.query_end;) {
            const u32 slot = query % NumQueries;
            const u32 count =
                static_cast<u32>(std::min<u64>(record.query_end - query, NumQueries - slot));
            device.resetQueryPool(*pool, slot, count);
            query += count;
        }
        query_read.store(std::max(first, record.query_end), std::memory_order_release);
        DumpSlot& slot = SlotFor(record.address);
        if (slot.address == record.address && slot.id == record.id) {
            slot.total = total;
            slot.lost = lost;
            slot.resolved = true;
        }
    }

    if (!complete) {
        return true;
    }
    for (u32 db = 0; db < record.num_dbs; ++db) {
        WriteGuest(record.address + u64(db) * record.stride,
                   ValidBit | ((total + db) / record.num_dbs));
    }
    return true;
}

} // namespace Vulkan
