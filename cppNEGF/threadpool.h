#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <atomic>
#include <functional>
#include <future>
#include <mutex>
#include <queue>


class threadpool
{
private:
    std::vector<std::future<void>> m_workers;

    std::queue<std::pair<std::function<void()>,std::promise<void>>> m_workQueue;
    std::mutex m_workQueueMutex;

    std::condition_variable m_newWork;
    std::atomic_bool m_toExit = false;

    friend class threadworker;
    friend void workFunction(threadpool*);

    threadpool(int num);

public:

    threadpool(threadpool& other) = delete;
    ~threadpool();

    std::future<void> queueWork(std::function<void()> work, bool dontQueue = false);
    static threadpool& getInstance(int num){static threadpool me(num); return me;};
};




#endif // THREADPOOL_H
