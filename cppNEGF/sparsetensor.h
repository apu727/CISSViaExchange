#ifndef SPARSETENSOR_H
#define SPARSETENSOR_H

#include "logger.h"

#include <cassert>
#include <string>
#include <vector>
#include <memory>




typedef int8_t noIdxType;
typedef int idxType;

namespace EinsumTemplates
{
using destType = std::integral_constant<int, 0>;
using firstType = std::integral_constant<int, 1>;
using secondType = std::integral_constant<int, 2>;

template <noIdxType noIndexes,noIdxType secondNoIndexes,noIdxType destNoIndexes, noIdxType repeatedNoIndexes, size_t stringLength,
         typename retOption, noIdxType retSize>
constexpr std::array<noIdxType,retSize>  parseEinsum(const char* einsumString);

template <noIdxType noIndexes,noIdxType secondNoIndexes, noIdxType destNoIndexes, size_t strLength>
class einsumState // helper not used in fast code
{
public:
    static constexpr const noIdxType repeatedNoIndexes = (noIndexes+secondNoIndexes-destNoIndexes)/2;
    const std::array<noIdxType,repeatedNoIndexes> repeatedIndexesFirstIndex;
    const std::array<noIdxType,repeatedNoIndexes> repeatedIndexesSecondIndex;
    const std::array<noIdxType,destNoIndexes>  destIndexesIndex;

    idxType destIndexesPos[destNoIndexes] = {};
    idxType firstIndexesPos[noIndexes] = {};
    idxType secondIndexesPos[secondNoIndexes] = {};
    idxType repeatedIndexesPos[repeatedNoIndexes] = {};

    constexpr einsumState(const char (&einsumString)[strLength]):
        repeatedIndexesFirstIndex(EinsumTemplates::parseEinsum<noIndexes,secondNoIndexes,destNoIndexes,repeatedNoIndexes,strLength-1,EinsumTemplates::firstType,repeatedNoIndexes>
                                  (einsumString)),
        repeatedIndexesSecondIndex(EinsumTemplates::parseEinsum<noIndexes,secondNoIndexes,destNoIndexes,repeatedNoIndexes,strLength-1,EinsumTemplates::secondType,repeatedNoIndexes>
                                   (einsumString)),
        destIndexesIndex(EinsumTemplates::parseEinsum<noIndexes,secondNoIndexes,destNoIndexes,repeatedNoIndexes,strLength-1,EinsumTemplates::destType,destNoIndexes>
                         (einsumString))
    {
        // for (noIdxType i = 0; i < repeatedNoIndexes; i++)
        //     logger().log("",repeatedIndexesFirstIndex[i]);
        // for (noIdxType i = 0; i < repeatedNoIndexes; i++)
        //     logger().log("",repeatedIndexesSecondIndex[i]);
        // for (noIdxType i = 0; i < destNoIndexes; i++)
        //     logger().log("",destIndexesIndex[i]);

    }
    bool generateEinsumOrdering(idxType IndexSize)
    {
        //super end condition
        if (destIndexesPos[destNoIndexes-1] >= IndexSize)
            return false;

        //Check repeated indexes


        //Increment repeated index
        for(noIdxType i = 0; i < repeatedNoIndexes; i++)
        {
            repeatedIndexesPos[i]++;
            firstIndexesPos[repeatedIndexesFirstIndex[i]]++;
            secondIndexesPos[repeatedIndexesSecondIndex[i]]++;
            if (repeatedIndexesPos[i] >= IndexSize && i < repeatedNoIndexes-1/*dont overflow check the last one as this is our end condition*/)
            {
                repeatedIndexesPos[i] = 0;
                firstIndexesPos[repeatedIndexesFirstIndex[i]] = 0;
                secondIndexesPos[repeatedIndexesSecondIndex[i]] = 0;
                //Reset to zero and go to next element
            }
            else
                break; // incremented
        }
        if (repeatedIndexesPos[repeatedNoIndexes-1] < IndexSize)
            return true;
        //fall through



        for(noIdxType i = 0; i < repeatedNoIndexes; i++)
        {
            repeatedIndexesPos[i] = 0;
            firstIndexesPos[repeatedIndexesFirstIndex[i]] = repeatedIndexesPos[i];
            secondIndexesPos[repeatedIndexesSecondIndex[i]] = repeatedIndexesPos[i];
        }
        //Increment
        for(noIdxType i = 0; i < destNoIndexes; i++)
        {
            destIndexesPos[i]++;
            noIdxType idx = destIndexesIndex[i]; //running out of names
            if (idx < noIndexes)
            {
                firstIndexesPos[idx]++;
            }
            else
            {
                secondIndexesPos[idx-noIndexes]++;
            }

            if (destIndexesPos[i] >= IndexSize && i < destNoIndexes-1/*dont overflow check the last one as this is our end condition*/)
            {
                destIndexesPos[i] = 0;
                noIdxType idx = destIndexesIndex[i]; //running out of names
                if (idx < noIndexes)
                {
                    firstIndexesPos[idx]=0;
                }
                else
                {
                    secondIndexesPos[idx-noIndexes]=0;
                }
                //Reset to zero and go to next element
            }
            else
                break; // incremented
        }
        if (destIndexesPos[destNoIndexes-1] < IndexSize)
            return true;
        else
            return false;// end condition

    }

};
}






template<noIdxType noIndexes,typename dataType>
class baseSparseTensor
{
public:
    typedef idxType indexArrayType[noIndexes];

    class iterator
    {
    protected:
        iterator(){};
    public:
        virtual ~iterator(){};
        virtual dataType& get(indexArrayType &indexes) = 0;
        virtual bool operator ==(const iterator& other) const  = 0;
        virtual bool operator <(const iterator& other) const  = 0;
        virtual bool operator <=(const iterator& other) const  = 0;
        virtual iterator& operator++() = 0;
    };

    class Constiterator
    {
    protected:
        Constiterator(){};
    public:
        virtual ~Constiterator(){};
        virtual const dataType get(indexArrayType &indexes) const = 0; // cant pass back a const reference since for the sparse view on a sparse matrix coeff ref would insert an element
        virtual bool operator ==(const Constiterator& other) const  = 0;
        virtual bool operator <(const Constiterator& other) const  = 0;
        virtual bool operator <=(const Constiterator& other) const  = 0;
        virtual Constiterator& operator++() = 0;
    };

    baseSparseTensor(){};
    virtual ~baseSparseTensor(){};


    virtual void copy(const baseSparseTensor& other) = 0;
    virtual idxType getSize() const = 0;
    virtual void setSize(idxType indexSize) = 0; // should also set to zero
    virtual dataType coeff(const indexArrayType &indexes, noIdxType nodeListToSearch = 0) const = 0;
    virtual dataType& coeffRef(const indexArrayType &indexes,noIdxType nodeListToSearch = 0) = 0;
    virtual std::string toString(noIdxType indexToOrderBy = 0) const = 0;
    virtual std::shared_ptr<iterator> begin() = 0;
    virtual std::shared_ptr<Constiterator> begin() const = 0;
    virtual std::shared_ptr<iterator> end() = 0;
    virtual std::shared_ptr<Constiterator> end() const = 0;
    virtual bool isEqual(const baseSparseTensor& other, numType tolerance = 0, bool printFailedIndex = false) const
    {
        auto myItStart = begin();
        auto myItEnd = end();
        auto otherItStart = other.begin();
        auto otherItEnd = other.end();
        indexArrayType currIdxs;
        while(*myItStart < *myItEnd)
        {
            dataType myVal = myItStart->get(currIdxs);
            if (std::abs(other.coeff(currIdxs) - myVal) > tolerance)
            {
                if(printFailedIndex)
                    logger().log("Comparison failed on ", currIdxs);
                return false;
            }
            ++*myItStart;
        }
        while(*otherItStart < *otherItEnd)
        {
            dataType otherVal = otherItStart->get(currIdxs);
            if (std::abs(coeff(currIdxs) - otherVal) > tolerance)
            {
                if(printFailedIndex)
                    logger().log("Comparison failed on ", currIdxs);
                return false;
            }
            ++*otherItStart;
        }
        return true;
    }
    bool m_isEinsumPrepared = false; // are we prepared for einsums?
    std::string m_einsumStringPrepared; // are we prepared for this einsum?
    int m_einsumPreparedPosition; // are we in the position we expected?



    constexpr noIdxType getNoIndexes() {return noIndexes;}
};

template <noIdxType noIndexes,typename dataType>
class DenseTensor;

template <noIdxType noIndexes,typename dataType>
class SparseTensor : public baseSparseTensor<noIndexes,dataType>
{// Everything is thread safe except things that may cause reallocations i.e. adding or removing elements, unless the number of elements has been specified in advance
public:
    typedef typename baseSparseTensor<noIndexes,dataType>::indexArrayType indexArrayType;
    friend DenseTensor<noIndexes,dataType>;
private:

    struct ElemType
    {
        indexArrayType m_idxs = {};
        dataType m_val;
        bool is(const indexArrayType &idxs) const
        {
            for (int i = 0; i < noIndexes; i++)
            {
                if (idxs[i] != m_idxs[i])
                    return false;
            }
            return true;
        }
        ElemType(const indexArrayType &idxs,dataType val = dataType())
        {
            memcpy(m_idxs,idxs,noIndexes*sizeof(typeof(*m_idxs)));
            m_val = val;
        }
    };

    struct node
    {
        idxType ElementIdx = -1;
        node* next = nullptr;
        node(){};
        ~node(){if (next != nullptr) delete next;}
    };

    std::vector<ElemType> m_elements; // stores the elements
    node** m_indexPtrs[noIndexes] = {nullptr}; // IndexPtrs[0] = array of ptrs to linked list of nodes. Each element in the list IndexPtrs[0][i] points to an element which has index0 = i
    node** m_nodeListEndPtrs[noIndexes] = {nullptr}; // array of pointers to the end of the linked list of nodes


    const ElemType* findElement(const indexArrayType &indexes, noIdxType nodeListToSearch = 0) const // finds the element with the specified indexes or nullptr, no error checking so private function
    {
        node** nodeList = m_indexPtrs[nodeListToSearch]; // list of starting points for index nodeListToSearch
        node* startNode = nodeList[indexes[nodeListToSearch]];
        if (startNode == nullptr)
            return nullptr; // not present
        while (startNode != nullptr) // traverse the linked list searching for the element.
        {
            idxType elementIdx = startNode->ElementIdx;
            if (m_elements[elementIdx].is(indexes))
                return &m_elements[elementIdx];
            startNode = startNode->next;
        }
        return nullptr;
    }

    ElemType* findElement(const indexArrayType &indexes, noIdxType nodeListToSearch = 0) // finds the element with the specified indexes or nullptr, no error checking so private function
    {
        node** nodeList = m_indexPtrs[nodeListToSearch]; // list of starting points for index nodeListToSearch
        node* startNode = nodeList[indexes[nodeListToSearch]];
        if (startNode == nullptr)
            return nullptr; // not present
        while (startNode != nullptr) // traverse the linked list searching for the element.
        {
            idxType elementIdx = startNode->ElementIdx;
            if (m_elements[elementIdx].is(indexes))
                return &m_elements[elementIdx];
            startNode = startNode->next;
        }
        return nullptr;
    }
    // ElemType* makeNewElement(const indexArrayType &indexes)
    // {
    //     assert(findElement(indexes) == nullptr);

    //     idxType elementIdx = m_elements.size();
    //     m_elements.emplace_back(indexes);
    //     for (noIdxType i = 0; i < noIndexes; i++)
    //     {
    //         node** nodeList = m_indexPtrs[i];
    //         node* startNode = nodeList[indexes[i]];

    //         if (startNode == nullptr)
    //         {
    //             nodeList[indexes[i]] = new node;
    //             nodeList[indexes[i]]->ElementIdx = elementIdx;
    //             continue;
    //         }
    //         while(startNode->next != nullptr)
    //             startNode = startNode->next;
    //         startNode->next = new node;
    //         startNode->next->ElementIdx = elementIdx;
    //     }
    //     return &m_elements[elementIdx];
    // }

    ElemType* makeNewElement(const indexArrayType &indexes)
    {
        assert(findElement(indexes) == nullptr);

        idxType elementIdx = m_elements.size();
        m_elements.emplace_back(indexes);
        for (noIdxType i = 0; i < noIndexes; i++)
        {
            node** nodeList = m_indexPtrs[i];
            node* startNode = m_nodeListEndPtrs[i][indexes[i]];

            if (startNode == nullptr)
            {
                nodeList[indexes[i]] = new node;
                nodeList[indexes[i]]->ElementIdx = elementIdx;
                m_nodeListEndPtrs[i][indexes[i]] = nodeList[indexes[i]];
                continue;
            }
            assert(startNode->next == nullptr);
            startNode->next = new node;
            startNode->next->ElementIdx = elementIdx;
            m_nodeListEndPtrs[i][indexes[i]] = startNode->next;
        }
        return &m_elements[elementIdx];
    }
protected:
    idxType m_indexSize = 0;

public:
    class SparseTensorConstIterator : public baseSparseTensor<noIndexes,dataType>::Constiterator
    {
        typedef typename baseSparseTensor<noIndexes,dataType>::Constiterator Constiterator;

        size_t m_size;
        size_t m_currIdx;
        const std::vector<ElemType> &m_elements;
        friend SparseTensor;
        public:
        SparseTensorConstIterator(size_t startIndex, const std::vector<ElemType> &elements ) : m_elements(elements)
        {
            m_size = m_elements.size();
            m_currIdx = startIndex;
        }
        ~SparseTensorConstIterator(){}
        virtual const dataType get(indexArrayType &indexes) const override
        {
            if (m_currIdx < m_size)
            {
                memcpy(indexes,m_elements[m_currIdx].m_idxs,sizeof(idxType)*noIndexes);
                return m_elements[m_currIdx].m_val;
            }
            return m_elements[0].m_val;
        };
        virtual bool operator ==(const Constiterator& other) const override
        {
            const SparseTensorConstIterator* otherPtr = dynamic_cast<const SparseTensorConstIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_currIdx == otherPtr->m_currIdx && &m_elements == &otherPtr->m_elements;

        }
        virtual bool operator <(const Constiterator& other) const override
        {
            const SparseTensorConstIterator* otherPtr = dynamic_cast<const SparseTensorConstIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_currIdx < otherPtr->m_currIdx && &m_elements == &otherPtr->m_elements;
        };
        virtual bool operator <=(const Constiterator& other) const  override
        {
            const SparseTensorConstIterator* otherPtr = dynamic_cast<const SparseTensorConstIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_currIdx <= otherPtr->m_currIdx && &m_elements == &otherPtr->m_elements;
        }
        virtual Constiterator& operator++() override
        {
            if (m_currIdx < m_size)
                m_currIdx++;
            return *this;
        }
    };

    class SparseTensorIterator : public baseSparseTensor<noIndexes,dataType>::iterator
    {
        typedef typename baseSparseTensor<noIndexes,dataType>::iterator iterator;

        size_t m_size;
        size_t m_currIdx;
        std::vector<ElemType> &m_elements;
        friend SparseTensor;
    public:
        SparseTensorIterator(size_t startIndex, std::vector<ElemType> &elements ) : m_elements(elements)
        {
            m_size = m_elements.size();
            m_currIdx = startIndex;
        }


        ~SparseTensorIterator(){}
        virtual dataType& get(indexArrayType &indexes) override
        {
            if (m_currIdx < m_size)
            {
                memcpy(indexes,m_elements[m_currIdx].m_idxs,sizeof(idxType)*noIndexes);
                return m_elements[m_currIdx].m_val;
            }
            return m_elements[0].m_val;
        };
        virtual bool operator ==(const iterator& other) const override
        {
            const SparseTensorIterator* otherPtr = dynamic_cast<const SparseTensorIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_currIdx == otherPtr->m_currIdx && &m_elements == &otherPtr->m_elements;

        }
        virtual bool operator <(const iterator& other) const override
        {
            const SparseTensorIterator* otherPtr = dynamic_cast<const SparseTensorIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_currIdx < otherPtr->m_currIdx && &m_elements == &otherPtr->m_elements;
        };
        virtual bool operator <=(const iterator& other) const  override
        {
            const SparseTensorIterator* otherPtr = dynamic_cast<const SparseTensorIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_currIdx <= otherPtr->m_currIdx && &m_elements == &otherPtr->m_elements;
        }
        virtual iterator& operator++() override
        {
            if (m_currIdx < m_size)
                m_currIdx++;
            return *this;
        }
    };
    SparseTensor(idxType indexSize = 0)
    {
        static_assert(noIndexes>0,"TENSOR MUST HAVE AT LEAST 1 INDEX");
        setSize(indexSize);
    }
    virtual ~SparseTensor()
    {
        setSize(0);
    }

    void copy(const baseSparseTensor<noIndexes,dataType>& other) override
    {
        setSize(other.getSize());
        auto it = other.begin();
        auto itEnd = other.end();
        while (*it < *itEnd)
        {
            indexArrayType indexes;
            dataType val;
            val = it->get(indexes);
            coeffRef(indexes) = val;
            ++(*it);
        }
    }

    idxType getSize() const override {return m_indexSize;}
    size_t getOccupiedSize() const {return m_elements.size();}
    void setSize(idxType indexSize) override
    {
        m_elements.clear();
        this->m_isEinsumPrepared = false;
        for (node** &nodeList : m_indexPtrs)
        {
            for (idxType i = 0; i < m_indexSize; i++)
            {
                if (nodeList[i] != nullptr)
                    delete nodeList[i];
            }
            if (nodeList != nullptr)
                delete[] nodeList;
            nodeList = nullptr;
        }

        for (node** &EndnodeList : m_nodeListEndPtrs)
        {
            //Dont need to delete the elements in the EndNodeList since they have already been deleted by nodeList
            if (EndnodeList != nullptr)
                delete[] EndnodeList;
            EndnodeList = nullptr;
        }

        m_indexSize = indexSize;
        assert(m_indexSize >=0);
        if (m_indexSize > 0)
        {
            for (node** &nodeList : m_indexPtrs)
            {
                if (nodeList != nullptr)
                {
                    logger().log("nodeList is not null when resizing?, memory leak?");
                }
                nodeList = new node*[m_indexSize];
                //default constructed nodes so already set to nullptr - according to standard, although it doesnt
                //memset(nodeList,0,m_indexSize*sizeof(node*)); - Not technically compliant since 0 =/= nullptr necessarily
                for (idxType i = 0; i < m_indexSize; i++)
                {
                    nodeList[i] = nullptr;// This will probably compile down to a memset anyway
                }
            }
            for (node** &EndnodeList : m_nodeListEndPtrs)
            {
                if (EndnodeList != nullptr)
                {
                    logger().log("EndnodeList is not null when resizing?, memory leak?");
                }
                EndnodeList = new node*[m_indexSize];
                //default constructed nodes so already set to nullptr - according to standard, although it doesnt
                //memset(nodeList,0,m_indexSize*sizeof(node*)); - Not technically compliant since 0 =/= nullptr necessarily
                for (idxType i = 0; i < m_indexSize; i++)
                {
                    EndnodeList[i] = nullptr;// This will probably compile down to a memset anyway
                }
            }
        }
    }
    dataType coeff(const indexArrayType &indexes, noIdxType nodeListToSearch = 0) const override// Returns the coefficient or dataType() if it doesnt exist
    {
        for (idxType index : indexes)
        {
            assert(index < m_indexSize && index >= 0);
        }
        assert(nodeListToSearch < noIndexes && nodeListToSearch >=0);
        const ElemType* elem = findElement(indexes,nodeListToSearch);
        if (elem == nullptr)
            return dataType();
        return elem->m_val;
    }

    dataType& coeffRef2(const indexArrayType &indexes,noIdxType nodeListToSearch = 0,bool guaranteedUnique = false)
    {
        for (idxType index : indexes)
        {
            assert(index < m_indexSize && index >= 0);
        }
        assert(nodeListToSearch < noIndexes && nodeListToSearch >=0);
        ElemType* elem = nullptr;
        if (!guaranteedUnique)
            elem = findElement(indexes,nodeListToSearch); // allocates if necessary
        if (elem == nullptr)
            elem = makeNewElement(indexes);
        return elem->m_val;
    }

    dataType& coeffRef(const indexArrayType &indexes,noIdxType nodeListToSearch = 0) override
    {
        return coeffRef2(indexes,nodeListToSearch);
    }

    std::string toString(noIdxType indexToOrderBy = 0) const override
    {
        assert(indexToOrderBy < noIndexes);
        std::ostringstream os;
        node** nodeList = m_indexPtrs[indexToOrderBy];
        for (idxType i = 0; i < m_indexSize; i++)
        {
            node* startNode = nodeList[i];
            while (startNode != nullptr)
            {
                const ElemType* elem = &m_elements[startNode->ElementIdx];
                os << '(';
                for (noIdxType j = 0; j < noIndexes; j++)
                {
                    os << elem->m_idxs[j] << ',';
                }
                os.seekp(-1,os.cur);
                os << "): " << elem->m_val << ' ';
                startNode = startNode->next;
            }
            if (nodeList[i] != nullptr)
                os << '\n';
        }
        return os.str();
    }

    template <noIdxType firstNoIndexes,noIdxType secondNoIndexes, noIdxType destNoIndexes, typename position,
             std::enable_if_t<std::is_same_v<position,EinsumTemplates::firstType>,bool> = true,size_t strLength>
    void prepareForEinsum(const char (&einsumString)[strLength], SparseTensor &other, bool alwaysUnique = false)
    {
        other.setSize(m_indexSize);
        EinsumTemplates::einsumState<firstNoIndexes,secondNoIndexes,destNoIndexes,strLength> state(einsumString);
        do
        {
            dataType val = coeff(state.firstIndexesPos);
            if (val != dataType())
            {
                if (alwaysUnique)
                    other.makeNewElement(state.firstIndexesPos)->m_val = val;
                else
                    other.coeffRef(state.firstIndexesPos) = val;
            }

        } while(state.generateEinsumOrdering(m_indexSize));
        other.m_isEinsumPrepared = true;
        other.m_einsumStringPrepared = einsumString;
        other.m_einsumPreparedPosition = 0;
    }

    template <noIdxType firstNoIndexes,noIdxType secondNoIndexes, noIdxType destNoIndexes, size_t strLength, typename position,
             std::enable_if_t<std::is_same_v<position,EinsumTemplates::secondType>,bool> = true>
    void prepareForEinsum(const char (&einsumString)[strLength], SparseTensor &other, bool alwaysUnique = false)
    {
        other.setSize(m_indexSize);
        EinsumTemplates::einsumState<firstNoIndexes,secondNoIndexes,destNoIndexes,strLength> state(einsumString);
        do
        {
            dataType val = coeff(state.secondIndexesPos);
            if (val != dataType())
            {
                if (alwaysUnique)
                    other.makeNewElement(state.secondIndexesPos)->m_val = val;
                else
                    other.coeffRef(state.secondIndexesPos) = val;
            }

        } while(state.generateEinsumOrdering(m_indexSize));
        other.m_isEinsumPrepared = true;
        other.m_einsumStringPrepared = einsumString;
        other.m_einsumPreparedPosition = 1;
    }

    virtual std::shared_ptr<typename baseSparseTensor<noIndexes,dataType>::iterator> begin() override
    {
        return std::make_shared<SparseTensorIterator>(0,m_elements);
    }
    virtual std::shared_ptr<typename baseSparseTensor<noIndexes,dataType>::iterator> end() override
    {
        return std::make_shared<SparseTensorIterator>(m_elements.size(),m_elements);
    }

    virtual std::shared_ptr<typename baseSparseTensor<noIndexes,dataType>::Constiterator> begin() const override
    {
        return std::make_shared<SparseTensorConstIterator>(0,m_elements);
    }
    virtual std::shared_ptr<typename baseSparseTensor<noIndexes,dataType>::Constiterator> end() const override
    {
        return std::make_shared<SparseTensorConstIterator>(m_elements.size(),m_elements);
    }


};


template<typename matrixType, typename dataType, bool isDiagonal>
class SparseTensorView : public baseSparseTensor<2,dataType>
{
    //typedef Eigen::Matrix<dataType,Eigen::Dynamic,Eigen::Dynamic> matrixType;
    typedef typename baseSparseTensor<2,dataType>::indexArrayType indexArrayType;
    matrixType& m_Matrix;
    idxType m_indexSize;
public:
    class SparseTensorViewIterator : public baseSparseTensor<2,dataType>::iterator
    {
        typedef typename baseSparseTensor<2,dataType>::iterator iterator;

        long m_iPos = 0;
        long m_jPos = 0;
        long m_iSize;
        long m_jSize;
        matrixType& m_Matrix;
        friend SparseTensorView;
        public:
        SparseTensorViewIterator(long iPos, long jPos, matrixType &matrix ) : m_Matrix(matrix)
        {
            m_iPos = iPos;
            m_jPos = jPos;
            m_iSize = matrix.rows();
            m_jSize = matrix.cols();
        }
        ~SparseTensorViewIterator(){}
        virtual dataType& get(indexArrayType &indexes) override
        {
            if (m_iPos < m_iSize && m_jPos < m_jSize)
            {
                indexes[0] = m_iPos;
                indexes[1] = m_jPos;
                return m_Matrix.coeffRef(m_iPos,m_jPos);
            }
            return m_Matrix.coeffRef(0,0);
        };
        virtual bool operator ==(const iterator& other) const override
        {
            const SparseTensorViewIterator* otherPtr = dynamic_cast<const SparseTensorViewIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_iPos == otherPtr->m_iPos && m_jPos == otherPtr->m_jPos && &m_Matrix == &otherPtr->m_Matrix;

        }
        virtual bool operator <(const iterator& other) const override
        {
            const SparseTensorViewIterator* otherPtr = dynamic_cast<const SparseTensorViewIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return ((m_iPos == otherPtr->m_iPos && m_jPos < otherPtr->m_jPos) || m_iPos < otherPtr->m_iPos) && &m_Matrix == &otherPtr->m_Matrix;
        };
        virtual bool operator <=(const iterator& other) const  override
        {
            const SparseTensorViewIterator* otherPtr = dynamic_cast<const SparseTensorViewIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return ((m_iPos == otherPtr->m_iPos && m_jPos <= otherPtr->m_jPos) || m_iPos <= otherPtr->m_iPos) && &m_Matrix == &otherPtr->m_Matrix;
        }
        virtual iterator& operator++() override
        {
            if (m_iPos < m_iSize)
            {
                m_jPos++;
                if (m_jPos >= m_jSize)
                {
                    m_jPos = 0;
                    m_iPos++;
                }
            }
            return *this;
        }
    };

    class SparseTensorViewConstIterator : public baseSparseTensor<2,dataType>::Constiterator
    {
        typedef typename baseSparseTensor<2,dataType>::Constiterator Constiterator;

        long m_iPos = 0;
        long m_jPos = 0;
        long m_iSize;
        long m_jSize;
        const matrixType& m_Matrix;
        friend SparseTensorView;
    public:
        SparseTensorViewConstIterator(long iPos, long jPos, const matrixType &matrix ) : m_Matrix(matrix)
        {
            m_iPos = iPos;
            m_jPos = jPos;
            m_iSize = matrix.rows();
            m_jSize = matrix.cols();
        }
        ~SparseTensorViewConstIterator(){}

        virtual const dataType get(indexArrayType &indexes) const override
        {
            if (m_iPos < m_iSize && m_jPos < m_jSize)
            {
                indexes[0] = m_iPos;
                indexes[1] = m_jPos;
                return m_Matrix.coeff(m_iPos,m_jPos);
            }
            return m_Matrix.coeff(0,0);
        };
        virtual bool operator ==(const Constiterator& other) const override
        {
            const SparseTensorViewConstIterator* otherPtr = dynamic_cast<const SparseTensorViewConstIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_iPos == otherPtr->m_iPos && m_jPos == otherPtr->m_jPos && &m_Matrix == &otherPtr->m_Matrix;

        }
        virtual bool operator <(const Constiterator& other) const override
        {
            const SparseTensorViewConstIterator* otherPtr = dynamic_cast<const SparseTensorViewConstIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return ((m_iPos == otherPtr->m_iPos && m_jPos < otherPtr->m_jPos) || m_iPos < otherPtr->m_iPos) && &m_Matrix == &otherPtr->m_Matrix;
        };
        virtual bool operator <=(const Constiterator& other) const  override
        {
            const SparseTensorViewConstIterator* otherPtr = dynamic_cast<const SparseTensorViewConstIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return ((m_iPos == otherPtr->m_iPos && m_jPos <= otherPtr->m_jPos) || m_iPos <= otherPtr->m_iPos) && &m_Matrix == &otherPtr->m_Matrix;
        }
        virtual Constiterator& operator++() override
        {
            if (m_iPos < m_iSize)
            {
                m_jPos++;
                if (m_jPos >= m_jSize)
                {
                    m_jPos = 0;
                    m_iPos++;
                }
            }
            return *this;
        }
    };

    SparseTensorView(matrixType& Matrix): m_Matrix(Matrix)
    {
        assert(Matrix.rows() == Matrix.cols());
        m_indexSize = Matrix.rows();
    }

    virtual void copy(const baseSparseTensor<2,dataType>& other) override
    {
        setSize(other.getSize());
        auto it = other.begin();
        auto itEnd = other.end();
        while (*it < *itEnd)
        {
            indexArrayType indexes;
            dataType val;
            val = it->get(indexes);
            if (val != dataType())
                m_Matrix.coeffRef(indexes[0],indexes[1]) = val;
            ++(*it);
        }

    };
    virtual idxType getSize() const override{return m_indexSize;}
    virtual void setSize(idxType indexSize) override
    {
        m_Matrix.resize(indexSize,indexSize,m_Matrix.getBasis(),m_Matrix.getSpinSym());
        m_Matrix.setZero();
        m_indexSize = indexSize;
    }
    virtual dataType coeff(const indexArrayType &indexes, noIdxType nodeListToSearch = 0) const override
    {
        if (isDiagonal && indexes[0] != indexes[1])
            return dataType();
        return m_Matrix.coeff(indexes[0],indexes[1]);
    }
    virtual dataType& coeffRef(const indexArrayType &indexes,noIdxType nodeListToSearch = 0) override
    {
        return m_Matrix.coeffRef(indexes[0],indexes[1]);
    };
    virtual std::string toString(noIdxType indexToOrderBy = 0) const override
    {
        std::ostringstream os;
        os << m_Matrix;
        return os.str();
    }

    virtual std::shared_ptr<typename baseSparseTensor<2,dataType>::iterator> begin() override
    {
        return std::make_shared<SparseTensorViewIterator>(0,0,m_Matrix);
    }

    virtual std::shared_ptr<typename baseSparseTensor<2,dataType>::Constiterator> begin() const override
    {
        return std::make_shared<SparseTensorViewConstIterator>(0,0,m_Matrix);
    }
    virtual std::shared_ptr<typename baseSparseTensor<2,dataType>::iterator> end() override
    {
        return std::make_shared<SparseTensorViewIterator>(0,m_indexSize,m_Matrix);
    }

    virtual std::shared_ptr<typename baseSparseTensor<2,dataType>::Constiterator> end() const override
    {
        return std::make_shared<SparseTensorViewConstIterator>(0,m_indexSize,m_Matrix);
    }



};

template <noIdxType noIndexes,typename dataType>
class DenseTensor : public baseSparseTensor<noIndexes,dataType>
{// Everything is thread safe except things that may cause reallocations i.e. adding or removing elements, unless the number of elements has been specified in advance
public:
    typedef typename baseSparseTensor<noIndexes,dataType>::indexArrayType indexArrayType;
private:
    dataType* m_Data = nullptr;
    size_t m_arrSize = 0; // annoying to compute

    const dataType* find(const indexArrayType &indexes) const // finds the element with the specified indexes, no error checking so private function
    {
        size_t offset = 0;
        for (noIdxType i = 0; i < noIndexes; i++)
        {
            offset = offset*m_indexSize + indexes[i];
        }
        return &(m_Data[offset]);
    }

    dataType* find(const indexArrayType &indexes) // finds the element with the specified indexes, no error checking so private function
    {
        size_t offset = 0;
        for (noIdxType i = 0; i < noIndexes; i++)
        {
            offset = offset*m_indexSize + indexes[i];
        }
        return &(m_Data[offset]);
    }
protected:
    idxType m_indexSize = 0;


public:
    class DenseTensorConstIterator : public baseSparseTensor<noIndexes,dataType>::Constiterator
    {
        typedef typename baseSparseTensor<noIndexes,dataType>::Constiterator Constiterator;

        size_t m_size;
        size_t m_currIdx;
        idxType m_indexSize;
        const dataType* m_Data;
    public:
        DenseTensorConstIterator(size_t startIndex, const dataType* data, idxType indexSize)
        {
            m_Data = data;
            m_indexSize = indexSize;
            m_size = 1;
            for (noIdxType i = 0; i< noIndexes; i++)
                m_size *= m_indexSize;
            m_currIdx = startIndex;
        }
        ~DenseTensorConstIterator(){}
        virtual const dataType get(indexArrayType &indexes) const override
        {
            size_t currIdx = m_currIdx;
            for (noIdxType i = 0; i < noIndexes; i++)
            {
                indexes[noIndexes - i-1] = currIdx % m_indexSize; //least significant index is the right hand most
                currIdx = currIdx/m_indexSize; //Integer divide
            }
            assert(currIdx == 0);
            assert(m_currIdx < m_size);
            return m_Data[m_currIdx];
        };
        virtual bool operator ==(const Constiterator& other) const override
        {
            const DenseTensorConstIterator* otherPtr = dynamic_cast<const DenseTensorConstIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return otherPtr->m_currIdx == m_currIdx || (m_currIdx >= m_size && otherPtr->m_currIdx >= m_size);
        }
        virtual bool operator <(const Constiterator& other) const override
        {
            const DenseTensorConstIterator* otherPtr = dynamic_cast<const DenseTensorConstIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_currIdx < otherPtr->m_currIdx;
        }
        virtual bool operator <=(const Constiterator& other) const  override
        {
            const DenseTensorConstIterator* otherPtr = dynamic_cast<const DenseTensorConstIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_currIdx <= otherPtr->m_currIdx;
        }
        virtual Constiterator& operator++() override
        {
            m_currIdx++;
            return *this;
        }
    };

    class DenseTensorIterator : public baseSparseTensor<noIndexes,dataType>::iterator
    {
        typedef typename baseSparseTensor<noIndexes,dataType>::iterator iterator;

        size_t m_size;
        size_t m_currIdx;
        idxType m_indexSize;
        dataType* m_Data;
    public:
        DenseTensorIterator(size_t startIndex, dataType* data, idxType indexSize)
        {
            m_Data = data;
            m_indexSize = indexSize;
            m_size = 1;
            for (noIdxType i = 0; i< noIndexes; i++)
                m_size *= m_indexSize;
            m_currIdx = startIndex;
        }


        ~DenseTensorIterator(){}
        virtual dataType& get(indexArrayType &indexes) override
        {
            size_t currIdx = m_currIdx;
            for (noIdxType i = 0; i < noIndexes; i++)
            {
                indexes[noIndexes - i-1] = currIdx % m_indexSize; //least significant index is the right hand most
                currIdx = currIdx/m_indexSize; //Integer divide
            }
            assert(currIdx == 0);
            assert(m_currIdx < m_size);
            return m_Data[m_currIdx];
        };
        virtual bool operator ==(const iterator& other) const override
        {
            const DenseTensorIterator* otherPtr = dynamic_cast<const DenseTensorIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return otherPtr->m_currIdx == m_currIdx || (m_currIdx >= m_size && otherPtr->m_currIdx >= m_size);
        }
        virtual bool operator <(const iterator& other) const override
        {
            const DenseTensorIterator* otherPtr = dynamic_cast<const DenseTensorIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_currIdx < otherPtr->m_currIdx;
        }
        virtual bool operator <=(const iterator& other) const  override
        {
            const DenseTensorIterator* otherPtr = dynamic_cast<const DenseTensorIterator*>(&other);
            if (otherPtr == nullptr)
                return false;
            return m_currIdx <= otherPtr->m_currIdx;
        }
        virtual iterator& operator++() override
        {
            m_currIdx++;
            return *this;
        }
    };
    DenseTensor(idxType indexSize = 0)
    {
        static_assert(noIndexes>0,"TENSOR MUST HAVE AT LEAST 1 INDEX");
        setSize(indexSize);
    }
    virtual ~DenseTensor()
    {
        setSize(0);
    }

    void copy(const baseSparseTensor<noIndexes,dataType>& other) override
    {
        setSize(other.getSize());
        auto it = other.begin();
        auto itEnd = other.end();
        while (*it < *itEnd)
        {
            indexArrayType indexes;
            dataType val;
            val = it->get(indexes);
            coeffRef(indexes) = val;
            ++(*it);
        }
    }

    idxType getSize() const override {return m_indexSize;}
    void setSize(idxType indexSize) override
    {
        assert(indexSize >= 0);
        if (m_indexSize != indexSize)
        {
            numType memSize = 8*pow(m_indexSize,noIndexes)/1e9;
            if (memSize > 8)
                logger().log("Deallocating Big Ram (GB):", memSize);

            m_indexSize = indexSize;
            m_arrSize = 1;
            for (noIdxType i = 0; i< noIndexes; i++)
                m_arrSize *= m_indexSize;

            if (m_Data != nullptr)
            {
                delete[] m_Data;
                m_Data = nullptr;
            }

            memSize = 8*pow(m_indexSize,noIndexes)/1e9;
            if (memSize > 8)
                logger().log("Allocating Big Ram (GB):", memSize);
            if (m_arrSize > 0)
                m_Data = new dataType[m_arrSize];
        }
        for (size_t i = 0; i < m_arrSize; i++)
        {
            m_Data[i] = dataType(); // will compile to a memset but in principle can be set to non-zero as a default for more complicated objects
        }
    }
    dataType coeff(const indexArrayType &indexes, noIdxType nodeListToSearch = 0) const override// Returns the coefficient or dataType() if it doesnt exist
    {
        for (noIdxType i = 0; i < noIndexes; i++)
        {
            assert(indexes[i] < m_indexSize);
        }
        return *find(indexes);
    }

    dataType& coeffRef(const indexArrayType &indexes,noIdxType nodeListToSearch = 0) override
    {
        for (noIdxType i = 0; i < noIndexes; i++)
        {
            assert(indexes[i] < m_indexSize);
        }
        return *find(indexes);
    }

    std::string toString(noIdxType indexToOrderBy = 0) const override
    {
        assert(indexToOrderBy < noIndexes);
        if (indexToOrderBy != 0)
            logger().log("DenseTensor cannot reorder indexes for printing");
        std::ostringstream os;
        auto itStart = begin();
        auto itEnd = end();

        indexArrayType currIdxs;
        dataType val = itStart->get(currIdxs);

        idxType currTopLevelIndex = currIdxs[0];
        while (*itStart < *itEnd)
        {
            val = itStart->get(currIdxs);
            if (currTopLevelIndex != currIdxs[0])
            {
                currTopLevelIndex = currIdxs[0];
                os << '\n';
            }
            os << '(';
            for (noIdxType j = 0; j < noIndexes; j++)
            {
                os << currIdxs[j] << ',';
            }
            os.seekp(-1,os.cur);
            os << "): " << val << ' ';
            ++*itStart;
        }
        os << '\n';// new line at end of printout
        return os.str();
    }

    template <noIdxType firstNoIndexes,noIdxType secondNoIndexes, noIdxType destNoIndexes, typename position,
             std::enable_if_t<std::is_same_v<position,EinsumTemplates::firstType>,bool> = true,size_t strLength>
    void prepareForEinsum(const char (&einsumString)[strLength], SparseTensor<noIndexes, dataType> &other, bool alwaysUnique = false)
    {
        other.setSize(m_indexSize);
        EinsumTemplates::einsumState<firstNoIndexes,secondNoIndexes,destNoIndexes,strLength> state(einsumString);
        do
        {
            dataType val = coeff(state.firstIndexesPos);
            if (val != dataType())
            {
                if (alwaysUnique)
                    other.makeNewElement(state.firstIndexesPos)->m_val = val;
                else
                    other.coeffRef(state.firstIndexesPos) = val;
            }

        } while(state.generateEinsumOrdering(m_indexSize));
        other.m_isEinsumPrepared = true;
        other.m_einsumStringPrepared = einsumString;
        other.m_einsumPreparedPosition = 0;
    }

    template <noIdxType firstNoIndexes,noIdxType secondNoIndexes, noIdxType destNoIndexes, size_t strLength, typename position,
             std::enable_if_t<std::is_same_v<position,EinsumTemplates::secondType>,bool> = true>
    void prepareForEinsum(const char (&einsumString)[strLength], SparseTensor<noIndexes, dataType> &other, bool alwaysUnique = false)
    {
        other.setSize(m_indexSize);
        EinsumTemplates::einsumState<firstNoIndexes,secondNoIndexes,destNoIndexes,strLength> state(einsumString);
        do
        {
            dataType val = coeff(state.secondIndexesPos);
            if (val != dataType())
            {
                if (alwaysUnique)
                    other.makeNewElement(state.secondIndexesPos)->m_val = val;
                else
                    other.coeffRef(state.secondIndexesPos) = val;
            }

        } while(state.generateEinsumOrdering(m_indexSize));
        other.m_isEinsumPrepared = true;
        other.m_einsumStringPrepared = einsumString;
        other.m_einsumPreparedPosition = 1;
    }

    virtual std::shared_ptr<typename baseSparseTensor<noIndexes,dataType>::iterator> begin() override
    {
        return std::make_shared<DenseTensorIterator>(0,m_Data,m_indexSize);
    }
    virtual std::shared_ptr<typename baseSparseTensor<noIndexes,dataType>::iterator> end() override
    {
        return std::make_shared<DenseTensorIterator>(m_arrSize,m_Data,m_indexSize);
    }

    virtual std::shared_ptr<typename baseSparseTensor<noIndexes,dataType>::Constiterator> begin() const override
    {
        return std::make_shared<DenseTensorConstIterator>(0,m_Data,m_indexSize);
    }
    virtual std::shared_ptr<typename baseSparseTensor<noIndexes,dataType>::Constiterator> end() const override
    {
        return std::make_shared<DenseTensorConstIterator>(m_arrSize,m_Data,m_indexSize);
    }



};

#endif // SPARSETENSOR_H
