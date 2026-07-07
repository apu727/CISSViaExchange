#ifndef COMPLEXMATRIXBUFFER_H
#define COMPLEXMATRIXBUFFER_H

#include "logger.h"
#include <condition_variable>
#include <mutex>
#include <utility>

//#define SANITISEBUFFERACCESS // tries to sanitise use after frees by invalidating the buffer after returning it.

template<class T>
class MatrixBufferObject;

template<class T>
class MatrixBufferTest;

template<class T>
class MatrixBuffer
{

    friend class MatrixBufferObject<T>;
    friend class MatrixBufferTest<T>;


    const int bufferSize = 10000;
    MatrixBuffer()
    {
        m_buffer.resize(bufferSize);
#ifdef SANITISEBUFFERACCESS
        for (int i = 0; i < bufferSize; i++)
        {
            m_buffer[i].~T();
            void* buf = &m_buffer[i];
            memset(buf,-1,sizeof(m_buffer[i]));
        }
#endif
        m_inUse.resize(bufferSize,false);
    }
    ~MatrixBuffer()
    {
        for (int i = 0; i < bufferSize; i++)
        {
            if (m_inUse[i])
                logger().log("There is a buffer still in use when the MatrixBuffer destructor is called",i);
#ifdef SANITISEBUFFERACCESS
            if (!m_inUse[i])
            {
                new ((void*)&m_buffer[i]) T;
            }
#endif
        }
    }
    std::vector<T> m_buffer;
    std::vector<bool> m_inUse;

    std::mutex m_lock;
    std::condition_variable m_notify;
    std::atomic<int> usageCount = 0;

    void returnBuffer(int idx)
    {
        std::lock_guard<std::mutex> lock(m_lock);
        usageCount--;
        if (usageCount > 3*bufferSize/2)
            logger().log("Buffer Pressure:",usageCount);
        m_inUse[idx] = false;
#ifdef SANITISEBUFFERACCESS
        m_buffer[idx].~T();
        void* buf = &m_buffer[idx];
        memset(buf,-1,sizeof(m_buffer[idx]));
#endif
        m_notify.notify_one();
    };

    int getIndex()
    {
        std::unique_lock<std::mutex> lock(m_lock);

        auto it = std::find(m_inUse.begin(),m_inUse.end(),false);
        if (it == m_inUse.end())
        {
            logger().log("Buffer Full, likely deadlock\n");
            m_notify.wait(lock, [CONST_REF_CAPTURE(m_inUse)](){return std::find(m_inUse.begin(),m_inUse.end(),false) != m_inUse.end();}); // waits until predicate is true
            it = std::find(m_inUse.begin(),m_inUse.end(),false);
        }
        assert(it != m_inUse.end());
#ifdef SANITISEBUFFERACCESS
        new ((void*)&m_buffer[it - m_inUse.begin()]) T;
#endif
        *it = true;

        usageCount++;
        if (usageCount > 3*bufferSize/2)
            logger().log("Buffer Pressure:",usageCount);

        return it - m_inUse.begin();
    }

public:
    static MatrixBuffer& getInstance()
    {
        static MatrixBuffer instance;
        return instance;
    }
    // MatrixBufferObject<T> getAMatrix()
    // {
    //     int idx = getIndex();
    //     return MatrixBufferObject<T>(idx);
    // }

};
template<typename T>
class RefCount
{
    std::atomic<int>* m_ref = nullptr;
    T m_data;
    std::function<void(T)> m_deleter;
    void cleanup()
    {
        if (m_ref == nullptr)
            return;
        if (m_ref->fetch_sub(1) == 1)
        {
            assert(*m_ref == 0);
            if(m_ref)
                delete m_ref;
            if (m_deleter)
                m_deleter(m_data);

        }
    }
public:
    RefCount() {}
    RefCount(const T& data, std::function<void(T)> deleter) : m_data(data), m_deleter(deleter)
    {
        m_ref = new std::atomic<int>(1);
    }
    ~RefCount()
    {
        cleanup();
    }
    RefCount(const RefCount& other) { *this = other; }
    RefCount& operator=(const RefCount& other)
    {
        if (other.m_ref)
            ++*other.m_ref;
        cleanup();
        m_ref = other.m_ref;

        m_data = other.m_data;
        m_deleter = other.m_deleter;
        return *this;
    }

    RefCount( RefCount&& other)
    {        
        std::swap(m_ref,other.m_ref);
        std::swap(m_data,other.m_data);
        std::swap(m_deleter,other.m_deleter);
    }
    RefCount& operator=(RefCount&& other)
    {
        std::swap(m_ref,other.m_ref);
        std::swap(m_data,other.m_data);
        std::swap(m_deleter,other.m_deleter);
        return *this;
    }
    const T& getData()const {return m_data;}
    int getCount() const {return  (m_ref == nullptr ? -1 : (int)*m_ref );}
    bool isEmpty() const {return m_ref == nullptr;}
};

template<class T>
class MatrixBufferObject
{
    friend class MatrixBuffer<T>;
    friend class MatrixBufferTest<T>;
    // MatrixBufferObject(int i)
    // {// Assumes we are the only one responsible for this
    //     m_refCount = RefCount<int>(i,ReturnBuffer);
    // }
    MatrixBufferObject(const RefCount<int>& r) {m_refCount = r;}
    RefCount<int> m_refCount;
    static void ReturnBuffer(int i){MatrixBuffer<T>::getInstance().returnBuffer(i);}
public:
    MatrixBufferObject()
    {
        int idx = MatrixBuffer<T>::getInstance().getIndex();
        m_refCount = RefCount<int>(idx,ReturnBuffer);
    }
    static MatrixBufferObject getEmpty(){return MatrixBufferObject(RefCount<int>());} // gives an unallocated one
    bool isEmpty()const {return m_refCount.isEmpty();}

    void detach()
    {
        int idx = MatrixBuffer<T>::getInstance().getIndex();
        RefCount<int> newRef(idx,ReturnBuffer);
        MatrixBuffer<T>::getInstance().m_buffer[newRef.getData()] = MatrixBuffer<T>::getInstance().m_buffer[m_refCount.getData()];
        m_refCount = newRef;
    }
    operator T&() {return get();}
    T* operator->() {return &get();}
    T& operator*() {return get();}
    //Technically there could be a race condition that causes both containers to detach but this is fine if inefficient
    //There is a nasty race condition. Consider two threads both of which have a reference to
    // MatrixBufferObject<T> A;
    // MatrixBufferObject<T> B = A; // Increments ref count
    // Thread1: T& A1 = A.get(); // causes detach -> m_refCount.getData() changes. Let idx1 = m_refCount.getData()
    // Thread2: T& A2 = A.get(); // may cause detach depending on race condition. m_refCount.getData() may change Let idx2 = m_refCount.getData()
    // If thread 2 causes detach then A1 points to the memory specified by idx1. Because Thread 2 caused an erroneous detach, idx2 != idx1. idx2 is marked as in use. idx1 is not.
    // Thread1's use of A1 is now a use after free as it points into memory marked as freed.
    // A mutex would solve this specific problem but cannot solve the following problem:
    // MatrixBufferObject<T> A;
    // MatrixBufferObject<T> B = A; // Increments ref count
    // Thread1: const T& A1 = std::as_const(A).get(); // Does not cause detach, A1 points into shared memory
    // Thread2: T& A2 = A.get(); // causes detach, A2 points into a new copy. A1 points into B.
    // Thread2: B = MatrixBufferObject<T>; // B now changes and so A1 points into freed memory.
    // The basic problem here is that we do not know the lifetime of a reference.
    // Options are: Abandon implicit sharing and always copy
    // Don't allow references to shared objects from multiple threads. Const references are okay.
    // There is no way to detect the second one.

    T& get() {assert(!m_refCount.isEmpty()); if (m_refCount.getCount() > 1) detach(); return MatrixBuffer<T>::getInstance().m_buffer[m_refCount.getData()];}

    operator const T&() const {return get();}
    const T* operator->() const {return &get();}
    const T& operator*() const {return get();}
    const T& get() const {assert(!m_refCount.isEmpty()); return MatrixBuffer<T>::getInstance().m_buffer[m_refCount.getData()];}
    int getId() const {return m_refCount.getData();} // used for seeing if this is aliasing

};

#endif // COMPLEXMATRIXBUFFER_H
