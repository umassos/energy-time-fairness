//
// Created by Qianlin Liang on 3/20/23.
//

#ifndef EFAIR_SCHEDULER_H
#define EFAIR_SCHEDULER_H

#include <vector>
#include <list>
#include <string>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <map>
#include <tvm/runtime/device_api.h>

#include "executor/executor.h"
#include "util/chfreq.h"
#include "util/common.h"

namespace efair {
namespace scheduler {

    class EFairScheduler {
    public:
        enum TaskState {
            Submitted,
            Started,
            Finished
        };

        EFairScheduler(MicroSeconds total_quantum_size, double alpha, tvm::Device device);
        ~EFairScheduler() = default;

        Status load_model(const std::string model_path, const std::string profile_path, const EntityID eid,
                          const std::string freq, ModelID &mid);
        Status create_entity(Priority priority, EntityID &eid);
        Status set_input(const ModelID &mid, const std::string &key, const void *input_data, size_t size);
        Status set_entity_priority(const EntityID eid, const Priority priority);
        Status wait_task(const TaskID &tid);
        Status new_task(const ModelID mid, TaskID &tid);
        Status summary_task_by_model();
        Status export_task_data(const std::string &path);
        Status run();
        Status shutdown();

    private:

        struct Task {
            friend EFairScheduler;
        private:
            ModelID mid;
            EntityID eid;
            TaskID tid;
            KernelIdx kernel_idx;
            TaskState status;
            std::mutex lock;
            std::condition_variable cv;

            std::chrono::steady_clock::time_point submit_t, start_t, end_t;
            efair::MicroSeconds service_time;   // service time from profile
            efair::MicroJoule energy_used;      // energy usage from profile

        public:
            bool is_finished() const;
            Status get_response_time(efair::MicroSeconds &response_time) const;
            Status get_timestamp(std::vector<std::chrono::steady_clock::time_point> &timestamps) const;
        };

        struct Model {
            friend EFairScheduler;
        private:
            ModelID mid;
            EntityID eid;
            std::string freq;
            std::shared_ptr<executor::Executor> executor;
            size_t num_kernels;
            MilliWatt max_power;
            MilliWatt power;
        };

        struct ScheduleEntity {
            friend EFairScheduler;
        private:
            EntityID eid;
            VRuntime vruntime;
            size_t weight;
            std::mutex lock;
            std::list<std::shared_ptr<Task>> fcfs_queue;
            MilliWatt max_power;
            MilliWatt avg_power;
            MicroSeconds runtime;
            MicroSeconds sched_slice;
        };

        void loop_body(void);
        Status get_total_weight(size_t &ret_weight);
        Status get_entity_avg_power(EntityID eid, MilliWatt &ret_avg_power);
        Status compute_entity_schedule_slices();

        // attributes
        std::mutex task_pool_lock, sched_entities_lock, model_pool_lock, rb_tree_lock;
        ModelID model_cnt;
        TaskID task_cnt;
        EntityID entity_cnt;
        std::unique_ptr<std::thread> scheduler_thread;
        std::atomic_bool _shutdown;
        std::multimap<VRuntime, std::shared_ptr<ScheduleEntity>> rb_tree;
        size_t total_weight;
        MicroSeconds min_sched_unit = 1000;

        MicroSeconds total_quantum_size;
        double alpha;

        tvm::Device dev;
        std::unordered_map<ModelID, std::shared_ptr<Model>> model_pool;
        std::unordered_map<EntityID, std::shared_ptr<ScheduleEntity>> sched_entities;
        std::unordered_map<TaskID, std::shared_ptr<Task>> task_pool;

        util::FrequencyController fc;

        static const std::unordered_map<Priority, size_t> priority_map;

    public:
        Status get_task(const TaskID tid, std::shared_ptr<Task> &ret_task);
    };

}   // namespace scheduler
}   // namespace efair

#endif //EFAIR_SCHEDULER_H
