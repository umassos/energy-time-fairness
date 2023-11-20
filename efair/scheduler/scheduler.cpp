//
// Created by Qianlin Liang on 3/21/23.
//

#include <fstream>
#include "scheduler/scheduler.h"

namespace efair {
namespace scheduler {

    const std::unordered_map<Priority, size_t> EFairScheduler::priority_map = {
            {-20, 88761},
            {-19, 71755},
            {-18, 56483},
            {-17, 46273},
            {-16, 36291},
            {-15, 29154},
            {-14, 23254},
            {-13, 18705},
            {-12, 14949},
            {-11, 11916},
            {-10, 9548},
            {-9,  7620},
            {-8,  6100},
            {-7,  4904},
            {-6,  3906},
            {-5,  3121},
            {-4,  2501},
            {-3,  1991},
            {-2,  1586},
            {-1,  1277},
            {0,   1024},
            {1,   820},
            {2,   655},
            {3,   526},
            {4,   423},
            {5,   335},
            {6,   272},
            {7,   215},
            {8,   172},
            {9,   137},
            {10,  110},
            {11,  87},
            {12,  70},
            {13,  56},
            {14,  45},
            {15,  36},
            {16,  29},
            {17,  23},
            {18,  18},
            {19,  15}
    };

    Status EFairScheduler::Task::get_response_time(efair::MicroSeconds &response_time) const {
        if (!this->is_finished())
            return Status::Fail;

        response_time = std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count();
        return Status::Succeed;
    }

    bool EFairScheduler::Task::is_finished() const {
        return status == TaskState::Finished;
    }

    Status
    EFairScheduler::Task::get_timestamp(std::vector<std::chrono::steady_clock::time_point> &timestamps) const {
        if (!this->is_finished())
            return Status::Fail;

        if (timestamps.size() != 0)
            timestamps.clear();

        timestamps.push_back(submit_t);
        timestamps.push_back(start_t);
        timestamps.push_back(end_t);

        return Status::Succeed;
    }

    EFairScheduler::EFairScheduler(MicroSeconds total_quantum_size, double alpha, tvm::Device device) :
            total_quantum_size(total_quantum_size),
            alpha(alpha),
            dev(device),
            model_cnt(0),
            task_cnt(0),
            entity_cnt(0),
            _shutdown(true) {}

    Status
    EFairScheduler::load_model(const std::string model_path, const std::string profile_path, const EntityID eid,
                               const std::string freq, ModelID &mid) {

        std::shared_ptr<executor::Executor> executor(new executor::Executor(model_path, profile_path, dev));
        ModelID issued_mid;
        {
            std::unique_lock<std::mutex> lock(model_pool_lock);
            issued_mid = model_cnt;
            model_cnt++;
        }

        std::shared_ptr<Model> m(new Model);
        m->mid = issued_mid;
        m->eid = eid;
        m->freq = freq;
        m->executor = std::move(executor);
        RETURN_STATUS(m->executor->get_gpu_power(m->freq, m->power))
        RETURN_STATUS(m->executor->get_max_gpu_power(m->max_power))
        RETURN_STATUS(m->executor->get_num_kernels(m->num_kernels))

        sched_entities[eid]->max_power =
                sched_entities[eid]->max_power < m->max_power ? m->max_power : sched_entities[eid]->max_power;

        LOG(INFO) << "Loaded model ID <" << issued_mid << "> " << m->executor->model_name << " with max power "
                  << m->max_power << " mWatt";
        LOG(INFO) << "Model " << issued_mid << " execution frequency " << m->freq << " power " << m->power;

        model_pool.insert({issued_mid, std::move(m)});
        mid = issued_mid;

        RETURN_STATUS(get_entity_avg_power(eid, sched_entities[eid]->avg_power));
        LOG(INFO) << "Entity " << eid << " current average power " << sched_entities[eid]->avg_power;

        return Status::Succeed;
    }

    Status EFairScheduler::compute_entity_schedule_slices() {
        auto num_entities = rb_tree.size();
        if (num_entities == 0)
            return Status::Succeed;

        std::multimap<MicroJoule, std::shared_ptr<ScheduleEntity>> energy_profile;
        MicroSeconds remain_slices = total_quantum_size;

        for (const auto & [vruntime, entity] : rb_tree){
            double fraction = static_cast<double>(entity->weight) / total_weight;
            auto w = static_cast<double>(priority_map.at(0)) / entity->weight;

            entity->sched_slice = static_cast<MicroJoule>(fraction * alpha * total_quantum_size);
            MicroJoule energy_consumption = entity->avg_power * 1e-3 * entity->sched_slice * w;
            energy_profile.insert({energy_consumption, entity});
            remain_slices -= entity->sched_slice;
        }

        while (remain_slices > 0){
            auto amount = remain_slices > min_sched_unit ? min_sched_unit : remain_slices;
            auto min_entity_it = energy_profile.begin();
            auto min_entity = min_entity_it->second;
            auto w = static_cast<double>(priority_map.at(0)) / min_entity->weight;

            min_entity->sched_slice += amount;
            MicroJoule energy_consumption = min_entity->avg_power * 1e-3 * min_entity->sched_slice * w;

            energy_profile.erase(min_entity_it);
            energy_profile.insert({energy_consumption, min_entity});
            remain_slices -= amount;
        }

        return Status::Succeed;
    }

    Status EFairScheduler::create_entity(Priority priority, EntityID &eid) {
        if (priority_map.find(priority) == priority_map.end()) {
            LOG(ERROR) << "Cannot find priority level " << priority;
            return Status::NotFound;
        }

        size_t weight = priority_map.at(priority);
        EntityID issued_eid;
        {
            std::unique_lock<std::mutex> lock(sched_entities_lock);
            issued_eid = entity_cnt;
            entity_cnt++;
        }

        std::shared_ptr<ScheduleEntity> entity(new ScheduleEntity);
        entity->eid = issued_eid;
        entity->weight = weight;
        entity->max_power = 0;
        entity->avg_power = 0;
        entity->runtime = 0;
        entity->sched_slice = 0;

        LOG(INFO) << "Created schedule entity ID <" << issued_eid << "> with priority " << priority;

        sched_entities.insert({issued_eid, std::move(entity)});
        eid = issued_eid;

        return Status::Succeed;
    }

    Status
    EFairScheduler::set_input(const ModelID &mid, const std::string &key, const void *input_data, size_t size) {
        RETURN_STATUS(model_pool[mid]->executor->set_input(key, input_data, size))
        return Status::Succeed;
    }

    Status EFairScheduler::set_entity_priority(const EntityID eid, const Priority priority) {
        if (sched_entities.find(eid) == sched_entities.end() || priority_map.find(priority) == priority_map.end())
            return Status::NotFound;
        sched_entities[eid]->weight = priority_map.at(priority);
        return Status::Succeed;
    }

    Status EFairScheduler::wait_task(const TaskID &tid) {
        std::shared_ptr<Task> task = task_pool[tid];

        std::unique_lock<std::mutex> lock(task->lock);
        task->cv.wait(lock, [task] { return task->status == TaskState::Finished; });

        return Status::Succeed;
    }

    Status EFairScheduler::new_task(const ModelID mid, TaskID &tid) {
        TaskID issued_tid;
        {
            std::unique_lock<std::mutex> lock(task_pool_lock);
            issued_tid = task_cnt;
            task_cnt++;
        }
        auto target_entity_id = model_pool[mid]->eid;
        std::shared_ptr<Task> task(new Task);
        task->submit_t = std::chrono::steady_clock::now();
        task->status = TaskState::Submitted;
        task->eid = target_entity_id;
        task->mid = mid;
        task->tid = issued_tid;
        task->energy_used = 0;
        task->service_time = 0;
        task->kernel_idx = 0;

        {
            std::unique_lock<std::mutex> lock(sched_entities[target_entity_id]->lock);
            sched_entities[target_entity_id]->fcfs_queue.push_back(task);

            if (sched_entities[target_entity_id]->fcfs_queue.size() == 1) {
                std::unique_lock<std::mutex> tree_lock(rb_tree_lock);
                if (rb_tree.size() == 0) {
                    sched_entities[target_entity_id]->vruntime = 0;
                } else {
                    auto tree_item = rb_tree.begin();
                    sched_entities[target_entity_id]->vruntime = tree_item->first;
                }

                rb_tree.insert({sched_entities[target_entity_id]->vruntime, sched_entities[target_entity_id]});
                get_total_weight(total_weight);
                compute_entity_schedule_slices();
            }
        }

        task_pool.insert({issued_tid, task});
        tid = issued_tid;
        return Status::Succeed;
    }

    Status EFairScheduler::get_task(const TaskID tid, std::shared_ptr<Task> &ret_task) {
        if (task_pool.find(tid) == task_pool.end())
            return Status::NotFound;

        ret_task = task_pool.at(tid);
        return Status::Succeed;
    }

    Status EFairScheduler::get_entity_avg_power(efair::EntityID eid, efair::MilliWatt &ret_avg_power) {
        MilliWatt power_sum = 0;
        size_t cnt = 0;

        for (const auto &[mid, model]: model_pool){
            if (model->eid == eid){
                power_sum += model->power;
                cnt++;
            }
        }

        ret_avg_power = power_sum / cnt;
        return Status::Succeed;
    }

    Status EFairScheduler::get_total_weight(size_t &ret_weight) {
        ret_weight = 0;

        for (const auto &[key, entity]: rb_tree) {
            ret_weight += entity->weight;
        }
        return Status::Succeed;
    }

    void EFairScheduler::loop_body() {
        if (rb_tree.empty()) return;
//        std::unique_lock<std::mutex> tree_lock(rb_tree_lock);
        auto start_t = std::chrono::steady_clock::now();

        std::multimap<VRuntime, std::shared_ptr<ScheduleEntity>>::iterator cur_entity_it;
        {
            std::unique_lock<std::mutex> tree_lock(rb_tree_lock);
            cur_entity_it = rb_tree.begin();
        }

        auto cur_entity = cur_entity_it->second;
        MicroSeconds time_meter = 0;
        MicroJoule energy_meter = 0;

//        LOG(INFO) << "Choose entity " << cur_entity->eid << "candidates:";
//        for (const auto & [e_vruntime, e] : rb_tree){
//            LOG(INFO) << "EID " << e->eid << " vruntime: " << e_vruntime;
//         }

//        double fraction = static_cast<double>(cur_entity->weight) / total_weight;
//        MicroSeconds quantum_size = static_cast<MicroSeconds>(fraction * total_quantum_size);
//        MicroJoule bucket_size = alpha * quantum_size * cur_entity->max_power * 1e-3;
        MicroSeconds quantum_size = cur_entity->sched_slice;

//        LOG(INFO) << "Quantum size " << quantum_size << " Bucket size " << bucket_size << " fraction: " << fraction;

        MicroJoule energy_used;
        MicroSeconds time_used;
        std::string cur_freq;

        Model *model;

        while (!cur_entity->fcfs_queue.empty() && time_meter < quantum_size) {
//            auto debug_start_t = std::chrono::steady_clock::now();
            auto task = cur_entity->fcfs_queue.front();

            if (task->status == TaskState::Submitted) {
                task->start_t = std::chrono::steady_clock::now();
                task->status = TaskState::Started;
            }

            model = model_pool[task->mid].get();
            ASSERT_STATUS(fc.get_frequency(cur_freq));
            if (cur_freq != model->freq) {
//                auto chfreq_start_t = std::chrono::steady_clock::now();
                fc.set_cur_frequency(model->freq);
//                auto chfreq_dur = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - chfreq_start_t).count();
//                LOG(INFO) << "Change frequency takes " << chfreq_dur << " µs";
            }

            model->executor->execute_kernel(task->kernel_idx, model->freq, time_used, energy_used);

            time_meter += time_used;
            energy_meter += energy_used;
            task->kernel_idx += 1;
            task->service_time += time_used;
            task->energy_used += energy_used;

            if (task->kernel_idx == model->num_kernels) {
                model->executor->sync();
                task->end_t = std::chrono::steady_clock::now();
                task->status = TaskState::Finished;

                MicroSeconds response_time;
                ASSERT_STATUS(task->get_response_time(response_time));
                LOG(INFO) << "Finished task <" << task->tid << "> in " << response_time << " µs";

                {
                    std::unique_lock<std::mutex> lock(cur_entity->lock);
                    cur_entity->fcfs_queue.pop_front();
                }

                {
                    std::unique_lock<std::mutex> lock(task->lock);
                    task->cv.notify_all();
                }
            }

//            LOG(INFO) << "Energy meter " << energy_meter << "/" << bucket_size << " Time meter: " << time_meter << "/" << quantum_size;
//            model->executor->sync();
//            auto debug_dur = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - debug_start_t).count();
//            LOG(INFO) << "Running task " << task->tid << " Time " << debug_dur << " Time used " << time_used
//                      << " Time meter: " << time_meter;

        }

        if (model != nullptr){
            model->executor->sync();
        }

        {
            std::unique_lock<std::mutex> tree_lock(rb_tree_lock);
            std::unique_lock<std::mutex> lock(cur_entity->lock);

            rb_tree.erase(cur_entity_it);
            if (!cur_entity->fcfs_queue.empty()) {
//                auto norm_time_meter = static_cast<double>(time_meter) / quantum_size;
//                auto norm_energy_meter = static_cast<double>(energy_meter) / bucket_size;
//                auto norm_vruntime = norm_time_meter < norm_energy_meter ? norm_energy_meter : norm_time_meter;

                cur_entity->vruntime += static_cast<double>(time_meter) / quantum_size;

                rb_tree.insert({cur_entity->vruntime, cur_entity});
            } else {
                get_total_weight(total_weight);
                compute_entity_schedule_slices();
            }
        }

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_t).count();
        cur_entity->runtime += duration;
        LOG(INFO) << "Entity <" << cur_entity->eid << "> runtime: " << cur_entity->runtime << " µs "
                  << "Time used this quantum: " << duration << " µs "<< "Frequency: " << model->freq;

    }

    Status EFairScheduler::run() {
        if (scheduler_thread.get() != nullptr) {
            LOG(ERROR) << "The scheduler has ran.";
            return Status::Fail;
        }

        this->_shutdown.store(false);
        scheduler_thread.reset(new std::thread([this] {
            while (true) {
                this->loop_body();
                if (this->_shutdown.load()) return;
            }
        }));

        LOG(INFO) << "Scheduler started";
        return Status::Succeed;
    }

    Status EFairScheduler::shutdown() {
        _shutdown.store(true);

        LOG(INFO) << "Stopping scheduler...";
        fc.shutdown();
        scheduler_thread->join();

        LOG(INFO) << "Scheduler has stopped.";
        return Status::Succeed;
    }

    Status EFairScheduler::summary_task_by_model() {
        std::unique_lock<std::mutex> t_lock(task_pool_lock);

        std::map<ModelID, MicroSeconds> time_stat;
        std::map<ModelID, MicroJoule> energy_stat;

        for (auto const & [tid, task] : task_pool){
            if (task->status != TaskState::Finished)
                continue;

            if (time_stat.find(task->mid) == time_stat.end())
                time_stat.insert({task->mid, task->service_time});
            else
                time_stat[task->mid] += task->service_time;

            if (energy_stat.find(task->mid) == energy_stat.end())
                energy_stat.insert({task->mid, task->energy_used});
            else
                energy_stat[task->mid] += task->energy_used;
        }

        LOG(INFO) << "Time usage: ";
        for (const auto & [mid, t] : time_stat){
            LOG(INFO) << "Model# " << mid << ": " << t << " µs\t Frequency " << model_pool[mid]->freq;
        }

        LOG(INFO) << "Energy usage: ";
        for (const auto & [mid, e] : energy_stat){
            LOG(INFO) << "Model# " << mid << ": " << e << " µJ\t Frequency " << model_pool[mid]->freq;
        }

        return Status::Succeed;
    }

    Status EFairScheduler::export_task_data(const std::string &path) {
        std::ofstream out_file(path);

        if (!out_file.is_open()){
            LOG(ERROR) << "Cannot save task data to file " << path;
            exit(1);
        }

        // Find the minimum start time
        auto min_time = std::chrono::steady_clock::now();
        for (const auto & [tid, task]: task_pool){
            if (task->status == TaskState::Finished && task->start_t < min_time){
                min_time = task->start_t;
            }
        }

        out_file << "task_id,entity_id,model_id,start_t,end_t,service_time,energy_used\n";

        for (const auto & [tid, task]: task_pool){
            if (task->status == TaskState::Finished){
                out_file << task->tid << "," << task->eid << "," << task->mid << "," <<
                std::chrono::duration_cast<std::chrono::microseconds>(task->start_t-min_time).count() << "," <<
                std::chrono::duration_cast<std::chrono::microseconds>(task->end_t-min_time).count() << "," <<
                task->service_time << "," << task->energy_used << "\n";
            }
        }
        out_file.close();

        return Status::Succeed;
    }
}   // namespace scheduler
}   // namespace efair