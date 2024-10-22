#ifndef _InferPool_H_
#define _InferPool_H_

#include <iostream>
#include <thread>
#include <string>
#include <queue>
#include <vector>
#include <memory>
#include "ThreadPool.hpp"
#include "model_infer/YoloXModel.hpp"
namespace ipool
{
    /**
     * @param Model 模型类
     * @param inputType 输入类型
     * @param outputType 输出类型
     */
    template <typename Model, typename inputType, typename outputType>
    class InferPool
    {
    private:
        int thread_num_;
        std::string model_path_;
        std::string metadata_path_;
        std::string device_name_;
        float conf_threshold_;
        int id_;
        std::mutex idMtx, queueMtx;
        std::unique_ptr<ThreadPool> pool;
        std::queue<std::future<outputType>> futs;
        std::vector<std::shared_ptr<Model>> models;

        int getModelId();

    public:
        InferPool(const std::string model_path, std::string metadata_path, const std::string &device, float conf_threshold, int thread_num);
        bool init();
        bool put(inputType inputData);
        bool get(outputType outputData);
        ~InferPool();
    };

    template <typename Model, typename inputType, typename outputType>
    inline int InferPool<Model, inputType, outputType>::getModelId()
    {
        std::lock_guard<std::mutex> lock(idMtx);
        int modelId = id_ % thread_num_;
        id_++;
        return modelId;
    }

    /**
     * @brief 线程池构造函数
     * @param model_path 模型路径
     * @param thread_num 线程数量
     */
    template <typename Model, typename inputType, typename outputType>
    inline InferPool<Model, inputType, outputType>::InferPool(const std::string model_path, std::string metadata_path, const std::string &device, float conf_threshold, int thread_num)
    {
        this->model_path_ = model_path;
        this->metadata_path_ = metadata_path;
        this->device_name_ = device;
        this->conf_threshold_ = conf_threshold;
        this->thread_num_ = thread_num;
        this->id_ = 0;
    }

    /**
     * @brief 线程池初始化
     */
    template <typename Model, typename inputType, typename outputType>
    inline bool InferPool<Model, inputType, outputType>::init()
    {
        try
        {
            this->pool = std::make_unique<ThreadPool>(this->thread_num_);
            for (int i = 0; i < this->thread_num_; i++)
                models.push_back(std::make_shared<Model>(this->model_path_, this->device_name_, this->metadata_path_, this->conf_threshold_));
        }
        catch (const std::bad_alloc &e)
        {
            std::cout << "Out of memory: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    /**
     * @brief 任务入口
     */
    template <typename Model, typename inputType, typename outputType>
    inline bool InferPool<Model, inputType, outputType>::put(inputType inputData)
    {
        std::lock_guard<std::mutex> lock(queueMtx);
        futs.push(pool->submit(&Model::infer, models[this->getModelId()], inputData));
        return true;
    }

    /**
     * @brief 将任务队列结果全部取出，释放类
     */
    template <typename Model, typename inputType, typename outputType>
    inline InferPool<Model, inputType, outputType>::~InferPool()
    {
        while (!futs.empty())
        {
            outputType temp = futs.front().get();
            futs.pop();
        }
    }

    /**
     * @brief 任务结果获取出口
     */
    template <typename Model, typename inputType, typename outputType>
    inline bool InferPool<Model, inputType, outputType>::get(outputType outputData)
    {
        std::lock_guard<std::mutex> lock(queueMtx);
        if (futs.empty() == true)
            return false;
        outputData = futs.front().get();
        futs.pop();
        return true;
    }

} // namespace ipool

#endif // _InferPool_H_