#ifndef EINSUMTEMPLATE_H
#define EINSUMTEMPLATE_H

#include "sparsetensor.h"
#include <cassert>

#include <array>
#include <cstdint>
#include <type_traits>
/*
 *  Here be very scary dragons!
 */

//#define TestEinsum
#ifdef TestEinsum
#warning Compiling with TestEinsum
#endif

namespace EinsumTemplates
{


template<noIdxType destNoIndexes, noIdxType repeatedNoIndexes,
         typename retOption, std::enable_if_t<std::is_same_v<retOption,destType>,bool> = true>
constexpr std::array<noIdxType,destNoIndexes> decide(std::array<noIdxType,destNoIndexes> destIndexesIndex,
                                                      std::array<noIdxType,repeatedNoIndexes> repeatedIndexesFirstIndex,
                                                      std::array<noIdxType,repeatedNoIndexes> repeatedIndexesSecondIndex)
{
    return destIndexesIndex;
}

template<noIdxType destNoIndexes, noIdxType repeatedNoIndexes,
         typename retOption, std::enable_if_t<std::is_same_v<retOption,firstType>,bool> = true>
constexpr std::array<noIdxType,repeatedNoIndexes> decide(std::array<noIdxType,destNoIndexes> destIndexesIndex,
                                                          std::array<noIdxType,repeatedNoIndexes> repeatedIndexesFirstIndex,
                                                          std::array<noIdxType,repeatedNoIndexes> repeatedIndexesSecondIndex)
{
    return repeatedIndexesFirstIndex;
}

template<noIdxType destNoIndexes, noIdxType repeatedNoIndexes,
         typename retOption, std::enable_if_t<std::is_same_v<retOption,secondType>,bool> = true>
constexpr std::array<noIdxType,repeatedNoIndexes> decide(std::array<noIdxType,destNoIndexes> destIndexesIndex,
                                                          std::array<noIdxType,repeatedNoIndexes> repeatedIndexesFirstIndex,
                                                          std::array<noIdxType,repeatedNoIndexes> repeatedIndexesSecondIndex)
{
    return repeatedIndexesSecondIndex;
}



template <noIdxType noIndexes,noIdxType secondNoIndexes,noIdxType destNoIndexes, noIdxType repeatedNoIndexes, size_t stringLength,
         typename retOption, noIdxType retSize>
constexpr std::array<noIdxType,retSize>  parseEinsum(const char* einsumString)
{// ijkl,jk,li means sum over jk with free index il and store into dest as li.
// very limited error checking TODO else it wont compile to a constexpr

    std::array<noIdxType,destNoIndexes> destIndexesIndex = {-1};
    std::array<noIdxType,repeatedNoIndexes> repeatedIndexesFirstIndex = {-1};
    std::array<noIdxType,repeatedNoIndexes> repeatedIndexesSecondIndex = {-1};

    char firstIndexes[noIndexes] = {};
    char secondIndexes[secondNoIndexes] = {};
    char destIndexes[destNoIndexes] = {};

    size_t firstIndexesIdx = 0;

    size_t secondIndexesIdx = 0;

    size_t destIndexesIdx = 0;

    int8_t onTensor = 0;

    for (size_t i = 0; i < stringLength; i++)
    {
        const char indexChar = einsumString[i];
        if (indexChar == ',')
        {
            assert(onTensor < 2);
            onTensor++;
            continue;
        }
        if (onTensor == 0)
        {
            assert(firstIndexesIdx < noIndexes);
            firstIndexes[firstIndexesIdx++] = indexChar;
        }
        else if (onTensor == 1)
        {
            assert(secondIndexesIdx < secondNoIndexes);
            secondIndexes[secondIndexesIdx++] = indexChar;
        }
        else
        {
            assert(destIndexesIdx < destNoIndexes);
            destIndexes[destIndexesIdx++] = indexChar;
        }
    }
    assert(firstIndexesIdx == noIndexes && secondIndexesIdx == secondNoIndexes && destIndexesIdx == destNoIndexes);

    size_t repeatedIndexesFirstIndexIdx = 0;

    size_t repeatedIndexesSecondIndexIdx = 0;

    size_t destIndexesIndexCount = 0;

    for (noIdxType i = 0; i < noIndexes; i++)
    {
        char indexChar = firstIndexes[i];
        size_t secondPosIdx = 0;
        for (; secondPosIdx < secondNoIndexes; secondPosIdx++)
        {
            if (secondIndexes[secondPosIdx] == indexChar)
                break;
        }
        bool inSecond = secondPosIdx != secondNoIndexes;

        size_t destPos = 0;
        for (; destPos < destNoIndexes; destPos++)
        {
            if (destIndexes[destPos] == indexChar)
                break;
        }
        bool inDest = destPos != destNoIndexes;
        assert(inSecond || inDest);
        if (inSecond)
        {
            assert(repeatedIndexesFirstIndexIdx < repeatedNoIndexes);
            assert(repeatedIndexesSecondIndexIdx < repeatedNoIndexes);
            repeatedIndexesFirstIndex[repeatedIndexesFirstIndexIdx++] = i;
            repeatedIndexesSecondIndex[repeatedIndexesSecondIndexIdx++] = secondPosIdx;
        }
        else
        {// in dest
            assert(destPos < destNoIndexes);
            destIndexesIndex[destPos] = i;
            destIndexesIndexCount++;
        }

    }
    //Do the same for the second but if in first then we don't care
    for (noIdxType i = 0; i < secondNoIndexes; i++)
    {
        char indexChar = secondIndexes[i];

        size_t firstPos = 0;
        for (; firstPos < noIndexes; firstPos++)
        {
            if (firstIndexes[firstPos] == indexChar)
                break;
        }
        bool inFirst = firstPos != noIndexes;
        if (inFirst)
            continue;

        size_t destPos = 0;
        for (; destPos < destNoIndexes; destPos++)
        {
            if (destIndexes[destPos] == indexChar)
                break;
        }
        bool inDest = destPos != destNoIndexes;

        // in dest
        assert(inDest);
        assert(destPos < destNoIndexes);
        destIndexesIndex[destPos] = i+noIndexes; // encoding the first/secondness of the index
        destIndexesIndexCount++;
    }
    assert(destIndexesIndexCount == destNoIndexes);
    assert(repeatedIndexesFirstIndexIdx == repeatedNoIndexes);
    assert(repeatedIndexesSecondIndexIdx == repeatedNoIndexes);
    return decide<destNoIndexes,repeatedNoIndexes,retOption>
        (destIndexesIndex,repeatedIndexesFirstIndex,repeatedIndexesSecondIndex);

}




template <noIdxType noIndexes,typename dataType, noIdxType secondNoIndexes, typename secondDataType, noIdxType destNoIndexes, typename destDataType,noIdxType repeatedNoIndexes>
void doEinsum(std::array<noIdxType,destNoIndexes>  destIndexesIndex, std::array<noIdxType,repeatedNoIndexes> repeatedIndexesFirstIndex, std::array<noIdxType,repeatedNoIndexes> repeatedIndexesSecondIndex,
              const baseSparseTensor<noIndexes,dataType>* first, const baseSparseTensor<secondNoIndexes,secondDataType>* second, baseSparseTensor<destNoIndexes,destDataType>* dest)
{
    //Done parsing Now sum
    const idxType firstIndexSize = first->getSize();
    const idxType secondIndexSize = second->getSize();
    const idxType destIndexSize = std::max(firstIndexSize,secondIndexSize);
    assert(firstIndexSize == secondIndexSize && firstIndexSize == destIndexSize); // For simplicity for now
    const idxType IndexSize = firstIndexSize;

    dest->setSize(IndexSize);
    typename baseSparseTensor<destNoIndexes,destDataType>::indexArrayType destIndexesPos = {};
    typename baseSparseTensor<noIndexes,dataType>::indexArrayType firstIndexesPos = {};
    typename baseSparseTensor<secondNoIndexes,secondDataType>::indexArrayType  secondIndexesPos = {};

    while (true)
    {// loop over destIndexes

        if (destIndexesPos[destNoIndexes-1] >= IndexSize)
            break;
        destDataType val = destDataType();

        idxType repeatedIndexesPos[repeatedNoIndexes] = {}; // this one doesnt have a sparseTensor associated with it so no typedef
        for(noIdxType i = 0; i < repeatedNoIndexes; i++)
        {
            firstIndexesPos[repeatedIndexesFirstIndex[i]] = repeatedIndexesPos[i];
            secondIndexesPos[repeatedIndexesSecondIndex[i]] = repeatedIndexesPos[i];
        }
        noIdxType firstNodeListHint = repeatedIndexesFirstIndex[repeatedNoIndexes-1];
        noIdxType secondNodeListHint = repeatedIndexesSecondIndex[repeatedNoIndexes-1];
        while (true)
        {// loop over summed indexes
            //Check end condition
            if (repeatedIndexesPos[repeatedNoIndexes-1] >= IndexSize)
                break;

            static_assert(secondDataType(0) == secondDataType());
            static_assert(dataType(0) == dataType());
            dataType val2 = second->coeff(secondIndexesPos,secondNodeListHint);
            if (val2 != secondDataType(0))
                val += first->coeff(firstIndexesPos,firstNodeListHint)*val2;

            //Increment
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
        }
        if (val != destDataType())
            dest->coeffRef(destIndexesPos) = val;

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

    }
}

template <noIdxType noIndexes,typename dataType, noIdxType secondNoIndexes, typename secondDataType, noIdxType destNoIndexes, typename destDataType,noIdxType repeatedNoIndexes>
void firstOptimisedEinsum(std::array<noIdxType,destNoIndexes>  destIndexesIndex, std::array<noIdxType,repeatedNoIndexes> repeatedIndexesFirstIndex, std::array<noIdxType,repeatedNoIndexes> repeatedIndexesSecondIndex,
              const baseSparseTensor<noIndexes,dataType>* first, const baseSparseTensor<secondNoIndexes,secondDataType>* second, baseSparseTensor<destNoIndexes,destDataType>* dest)
{
    //Done parsing Now sum, first is already in the correct order
    const idxType firstIndexSize = first->getSize();
    const idxType secondIndexSize = second->getSize();
    const idxType destIndexSize = std::max(firstIndexSize,secondIndexSize);
    assert(firstIndexSize == secondIndexSize && firstIndexSize == destIndexSize); // For simplicity for now
    const idxType IndexSize = firstIndexSize;

    dest->setSize(IndexSize);
    typename baseSparseTensor<destNoIndexes,destDataType>::indexArrayType destIndexesPos = {};
    typename baseSparseTensor<noIndexes,dataType>::indexArrayType firstIndexesPos = {};
    typename baseSparseTensor<secondNoIndexes,secondDataType>::indexArrayType  secondIndexesPos = {};
    typename baseSparseTensor<noIndexes,dataType>::indexArrayType firstIndexesItPos = {};
    auto firstIt = first->begin();
    auto firstItEnd = first->end();

    noIdxType numberOfFirstIndexesInDestination = 0;
    noIdxType numberOfSecondIndexesInDestination = 0;
    noIdxType IndexOfHighestSecondIndexInDestination = 0; // the index in destIndexesPos that represents the slowest changing position.
    for(noIdxType i = 0; i < destNoIndexes; i++)
    {
        noIdxType idx = destIndexesIndex[i]; //running out of names
        if (idx < noIndexes)
        {
            numberOfFirstIndexesInDestination++;
        }
        else
        {
            numberOfSecondIndexesInDestination++;
            IndexOfHighestSecondIndexInDestination = i;
        }
    }

    while (true)
    {// loop over destIndexes

        if (destIndexesPos[IndexOfHighestSecondIndexInDestination] >= IndexSize)
            break;
        destDataType val = destDataType();

        idxType repeatedIndexesPos[repeatedNoIndexes] = {}; // this one doesnt have a sparseTensor associated with it so no typedef
        for(noIdxType i = 0; i < repeatedNoIndexes; i++)
        {
            firstIndexesPos[repeatedIndexesFirstIndex[i]] = repeatedIndexesPos[i];
            secondIndexesPos[repeatedIndexesSecondIndex[i]] = repeatedIndexesPos[i];
        }
        noIdxType firstNodeListHint = repeatedIndexesFirstIndex[repeatedNoIndexes-1];
        noIdxType secondNodeListHint = repeatedIndexesSecondIndex[repeatedNoIndexes-1];
        while (true)
        {// loop over summed indexes
            //Check end condition
            dataType val1 = firstIt->get(firstIndexesItPos);
            for(noIdxType i = 0; i < repeatedNoIndexes; i++)
            {
                if (i == repeatedNoIndexes-1 && repeatedIndexesPos[i] > firstIndexesItPos[repeatedIndexesFirstIndex[i]])
                    repeatedIndexesPos[i] = IndexSize; // dont allow the last one to decrease, this is a break condition if the firstIndexIt only contains repeated indexes
                else
                    repeatedIndexesPos[i] = firstIndexesItPos[repeatedIndexesFirstIndex[i]];
                secondIndexesPos[repeatedIndexesSecondIndex[i]] = repeatedIndexesPos[i];
                firstIndexesPos[repeatedIndexesFirstIndex[i]] = repeatedIndexesPos[i];
            }
            if(memcmp(firstIndexesItPos,firstIndexesPos,noIndexes*sizeof(firstIndexesPos[0])) != 0)
            {
                //This means we need to change destination indexes
                break;
            }

            //we found it so go to the next one, Since firstTensor contains at least one repeated index we know the next one will be different so can safely move on
            // std::cerr << "computing:";
            // for(noIdxType i = 0; i < noIndexes; i++)
            // {
            //     std::cerr << firstIndexesPos[i] << ',';
            // }
            // std::cerr << " with ";
            // for(noIdxType i = 0; i < secondNoIndexes; i++)
            // {
            //     std::cerr << secondIndexesPos[i] << ',';
            // }
            // std::cerr << " To make ";
            // for(noIdxType i = 0; i < destNoIndexes; i++)
            // {
            //     std::cerr << destIndexesPos[i] << ',';
            // }

            // std::cerr << '\n';

            ++(*firstIt);
            if (*firstIt == *firstItEnd)
                firstIt = first->begin(); // loop round, still in correct order?
            static_assert(secondDataType(0) == secondDataType());
            static_assert(dataType(0) == dataType());

            val += val1*second->coeff(secondIndexesPos,secondNodeListHint); // we assume getting second->coeff is quick e.g. dense matrix

        }
        if (val != destDataType())
            dest->coeffRef(destIndexesPos) = val;

        // Increment the destination via jumping to the next element that has the correct first indices ignoring any intermediate indices
        bool incremented = false;
        bool overflowed = false;
        for(noIdxType i = 0; i < destNoIndexes; i++)
        {
            noIdxType idx = destIndexesIndex[i]; //running out of names
            if (idx < noIndexes)
            {
                if (destIndexesPos[i] != firstIndexesItPos[idx])
                {
                    incremented = true;
                    if (destIndexesPos[i] > firstIndexesItPos[idx])
                        overflowed = true;
                    else
                        overflowed = false; // only need to know if the last changed firstIndex was an overflow
                    destIndexesPos[i] =  firstIndexesItPos[idx];
                    firstIndexesPos[idx] = destIndexesPos[i];
                }
            }
        }
        if ((overflowed || incremented == false) && numberOfSecondIndexesInDestination > 0)
        {//increment the next second index
            for(noIdxType i = 0; i < destNoIndexes; i++)
            {
                noIdxType idx = destIndexesIndex[i]; //running out of names
                if (idx >= noIndexes)
                {
                    destIndexesPos[i]++;
                    secondIndexesPos[idx-noIndexes]++;
                    if (destIndexesPos[i] >= IndexSize && i != IndexOfHighestSecondIndexInDestination) // overflow unless it is the end condition
                    {
                        destIndexesPos[i] = 0;
                        secondIndexesPos[idx-noIndexes] = 0;
                    }
                    else
                        break;
                }
            }
        }
        if (overflowed && numberOfSecondIndexesInDestination == 0)
            break;

    }
}

template <noIdxType noIndexes,typename dataType, noIdxType secondNoIndexes, typename secondDataType, noIdxType destNoIndexes, typename destDataType,noIdxType repeatedNoIndexes>
void decideHowToDoEinsum(std::string einsumString, std::array<noIdxType,destNoIndexes>  destIndexesIndex, std::array<noIdxType,repeatedNoIndexes> repeatedIndexesFirstIndex, std::array<noIdxType,repeatedNoIndexes> repeatedIndexesSecondIndex,
              const baseSparseTensor<noIndexes,dataType>* first, const baseSparseTensor<secondNoIndexes,secondDataType>* second, baseSparseTensor<destNoIndexes,destDataType>* dest)
{

    if (first->m_isEinsumPrepared && !(first->m_einsumPreparedPosition == 0 &&  first->m_einsumStringPrepared == einsumString))
    {
        logger().log("Einsum is prepared but cannot do quickly");
    }

    if (first->m_isEinsumPrepared && first->m_einsumPreparedPosition == 0 &&  first->m_einsumStringPrepared == einsumString)
    {
        firstOptimisedEinsum<noIndexes,dataType,secondNoIndexes,secondDataType,destNoIndexes,destDataType,repeatedNoIndexes>
            (destIndexesIndex,repeatedIndexesFirstIndex,repeatedIndexesSecondIndex,first,second,dest);
#ifdef TestEinsum
        SparseTensor<destNoIndexes,destDataType> destTest;
        doEinsum<noIndexes,dataType,secondNoIndexes,secondDataType,destNoIndexes,destDataType,repeatedNoIndexes>
            (destIndexesIndex,repeatedIndexesFirstIndex,repeatedIndexesSecondIndex,first,second,&destTest);
        releaseAssert(destTest.isEqual(*dest,0,true), "Einsum Testing Failed");
#endif
    }
    else
    {
        doEinsum<noIndexes,dataType,secondNoIndexes,secondDataType,destNoIndexes,destDataType,repeatedNoIndexes>
            (destIndexesIndex,repeatedIndexesFirstIndex,repeatedIndexesSecondIndex,first,second,dest);
    }
}

}


template <noIdxType noIndexes,typename dataType,noIdxType secondNoIndexes, typename secondDataType, noIdxType destNoIndexes, typename destDataType, size_t strLength>
constexpr auto
Einsum(const char (&einsumString)[strLength])
{
    const noIdxType repeatedNoIndexes = (noIndexes+secondNoIndexes-destNoIndexes)/2;
    const std::array<noIdxType,repeatedNoIndexes> repeatedIndexesFirstIndex =
        EinsumTemplates::parseEinsum<noIndexes,secondNoIndexes,destNoIndexes,repeatedNoIndexes,strLength-1,EinsumTemplates::firstType,repeatedNoIndexes>
        (einsumString);

    const std::array<noIdxType,repeatedNoIndexes> repeatedIndexesSecondIndex =
        EinsumTemplates::parseEinsum<noIndexes,secondNoIndexes,destNoIndexes,repeatedNoIndexes,strLength-1,EinsumTemplates::secondType,repeatedNoIndexes>
        (einsumString);

    const std::array<noIdxType,destNoIndexes>  destIndexesIndex =
        EinsumTemplates::parseEinsum<noIndexes,secondNoIndexes,destNoIndexes,repeatedNoIndexes,strLength-1,EinsumTemplates::destType,destNoIndexes>
        (einsumString);



    return [=](const baseSparseTensor<noIndexes,dataType>* first, const baseSparseTensor<secondNoIndexes,secondDataType>* second, baseSparseTensor<destNoIndexes,destDataType>* dest)
    {
        EinsumTemplates::decideHowToDoEinsum<noIndexes,dataType,secondNoIndexes,secondDataType,destNoIndexes,destDataType,repeatedNoIndexes>
            (einsumString,destIndexesIndex,repeatedIndexesFirstIndex,repeatedIndexesSecondIndex,first,second,dest);
    };
}







#endif // EINSUMTEMPLATE_H
