#include "complexmatrixbuffer.h"
#include "logger.h"

void log(const std::string &str, const std::atomic<int> &val)
{
    logger().log(str,val);
}

void log(const std::string &str)
{
    logger().log(str);
}
