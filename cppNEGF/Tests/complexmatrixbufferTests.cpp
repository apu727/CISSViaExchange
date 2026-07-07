#include "testutil.hpp"
#include <complexmatrixbuffer.h>
#include <threadpool.h>


template <class T>
class MatrixBufferTest
{
public:
    MatrixBuffer<T>& buf;
    MatrixBufferTest():buf(MatrixBuffer<T>::getInstance()){}
    ~MatrixBufferTest(){}

    int checkBuffers()
    {   int ret = 0;
        ret += expect((size_t)buf.bufferSize,buf.m_buffer.size(),__PRETTY_FUNCTION__,"Buffer Size wrong");
        ret += expect((size_t)buf.bufferSize,buf.m_inUse.size(),__PRETTY_FUNCTION__,"Buffer Size wrong");
        for (bool b : buf.m_inUse)
            ret += expect(false,b,__PRETTY_FUNCTION__,"Buffer uninitialised");
        return ret;
    }
    int checkGetBuffers()
    {
        using namespace std::chrono_literals;

        int ret = 0;
        auto& pool = threadpool::getInstance(NUM_CORES);
        std::vector<std::future<void>> futures;
        for (int count = 0; count < 4*buf.bufferSize; count++)
        {
            futures.push_back(pool.queueWork([](){MatrixBufferObject<T>();}));
        }
        for (auto& fut: futures)
            ret += expect((int)std::future_status::ready,(int)fut.wait_for(1s),__PRETTY_FUNCTION__,"Buffer Timeout");
        return ret;
    }

    int checkMoveBuffers()
    {
        using namespace std::chrono_literals;
        int ret = 0;
        std::vector<MatrixBufferObject<T>> bufs;
        std::vector<MatrixBufferObject<T>> bufsMoved;
        bufs.resize(buf.bufferSize);
        ret += expect(buf.bufferSize,(int)buf.usageCount,__PRETTY_FUNCTION__,"Usage count error");

        for (int count = 0; count < buf.bufferSize; count++)
            bufsMoved.push_back(MatrixBufferObject<T>::getEmpty());
        ret += expect(buf.bufferSize,(int)buf.usageCount,__PRETTY_FUNCTION__,"Usage count error2");


        for (int count = 0; count < buf.bufferSize; count++)
            bufsMoved[count] = std::move(bufs[count]);
        ret += expect(buf.bufferSize,(int)buf.usageCount,__PRETTY_FUNCTION__,"Usage count error3");
        bufs.clear();
        ret += expect(buf.bufferSize,(int)buf.usageCount,__PRETTY_FUNCTION__,"Usage count error4");
        bufsMoved.clear();
        ret += expect(0,(int)buf.usageCount,__PRETTY_FUNCTION__,"Usage count error5");
        return ret;
    }
    int checkBuffersUnique()
    {
        using namespace std::chrono_literals;
        int ret = 0;
        auto& pool = threadpool::getInstance(NUM_CORES);
        std::vector<std::future<void>> futures;

        std::vector<MatrixBufferObject<T>> bufs;
        for (int count = 0; count < buf.bufferSize; count++)
            bufs.push_back(MatrixBufferObject<T>::getEmpty());


        for (int count = 0; count < buf.bufferSize; count++)
        {
            futures.push_back(pool.queueWork([BY_REF_CAPTURE(bufs),BY_VAL_CAPTURE(count)](){bufs[count] = MatrixBufferObject<T>();}));
        }
        for (auto& fut: futures)
            ret += expect((int)std::future_status::ready,(int)fut.wait_for(1s),__PRETTY_FUNCTION__,"Buffer Timeout");
        std::vector<int> idxs;
        for (auto& b:bufs)
        {
            auto it = std::find(idxs.begin(),idxs.end(),b.m_refCount.getData());
            ret += expect(idxs.end()-idxs.begin(),it-idxs.begin(),__PRETTY_FUNCTION__,"Idx clash");
            idxs.push_back(b.m_refCount.getData());
        }
        return ret;
    }
    int checkDetach()
    {
        int ret = 0;
        MatrixBufferObject<T> first;
        first->resize(2,2);
        first->setZero();
        first->coeffRef(0,0) = 1;
        MatrixBufferObject<T> second = first;
        const MatrixBufferObject<T>& secondRef = second;
        ret += expect(2,first.m_refCount.getCount(),__PRETTY_FUNCTION__,"RefCount");
        ret += expect(complexType(1.),secondRef->coeff(0,0),__PRETTY_FUNCTION__,"const ref value");

        ret += expect(2,first.m_refCount.getCount(),__PRETTY_FUNCTION__,"RefCount2");
        second->coeffRef(0,0) = 2;
        ret += expect(complexType(2.),secondRef->coeff(0,0),__PRETTY_FUNCTION__,"detach modify");
        ret += expect(complexType(1.),first->coeff(0,0),__PRETTY_FUNCTION__,"detach modify 2");
        ret += expect(1,first.m_refCount.getCount(),__PRETTY_FUNCTION__,"RefCount3");
        ret += expect(1,second.m_refCount.getCount(),__PRETTY_FUNCTION__,"RefCount4");

        //Check use after free
        first = first;
        ret += expect(1,first.m_refCount.getCount(),__PRETTY_FUNCTION__,"RefCount5");
        first = std::move(first);
        return ret;
    }

    int threadedDetach()
    {
        /* this explicitly doesnt pass. It is a race condition that should error out when using the sanitiser*/
        int ret = 0;
        int count = 0;
        while (count < 100000)
        {
            MatrixBufferObject<T> first;
            first->resize(2,2);
            first->setZero();
            first->coeffRef(0,0) = 1;
            MatrixBufferObject<T> second = first;

            std::mutex syncMutex;
            std::atomic_int count = 0;
            auto work = [&](int idx)
            {
                T& A = first.get();// if the two detaches are called simultaneously this can point into `unallocated' memory
                count++;
                while (true)
                {
                    if (count == 2)
                        break;
                }
                std::unique_lock<std::mutex> lock(syncMutex);
                A.coeffRef(0,idx) += 1;
            };
            threadpool& pool = threadpool::getInstance(NUM_CORES);
            if (NUM_CORES < 2)
            {
                logger().log("Test cannot be done without multithreading");
                return 1;
            }
            std::future<void> fut1 = pool.queueWork([CONST_REF_CAPTURE(work)](){work(0);});
            std::future<void> fut2 = pool.queueWork([CONST_REF_CAPTURE(work)](){work(1);});

            fut1.wait();
            fut2.wait();
            ret += expect(complexType(2),first->coeffRef(0,0),__PRETTY_FUNCTION__,"threaded Detach race condition, Expected!!!!");
            ret += expect(complexType(1),first->coeffRef(0,1),__PRETTY_FUNCTION__,"threaded Detach race condition, Expected!!!!");

        }
        return ret;
    }

};


int complexmatrixbufferTests(int argc, char** argv)
{
    int ret = 0;
    MatrixBufferTest<Eigen::SparseMatrix<complexType>> sparseTest;
    ret += sparseTest.checkBuffers();
    ret += sparseTest.checkGetBuffers();
    ret += sparseTest.checkMoveBuffers();
    ret += sparseTest.checkBuffersUnique();
    ret += sparseTest.checkDetach();
    // ret += (sparseTest.threadedDetach() == 0); // should error

    MatrixBufferTest<Eigen::MatrixXcd> denseTest;
    ret += denseTest.checkBuffers();
    ret += denseTest.checkGetBuffers();
    ret += denseTest.checkMoveBuffers();
    ret += denseTest.checkBuffersUnique();
    ret += denseTest.checkDetach();
    // ret += (denseTest.threadedDetach() == 0);
    return ret > 0 ? -1 : 0;
}

