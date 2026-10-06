#pragma once

#include <cstddef>
#include <functional>

namespace plegl {

// Splits [begin, end) into chunks of at least `grain` items and runs `fn(chunkBegin, chunkEnd)`
// on a shared worker pool. Blocks until every chunk is done. Small ranges run inline.
//
// This is the only parallel primitive the engine uses, so the backing pool can be swapped
// (for example to oneTBB) without touching callers.
void parallelFor(std::size_t begin, std::size_t end, std::size_t grain,
                 const std::function<void(std::size_t, std::size_t)>& fn);

// Number of threads the pool uses, including the calling thread.
std::size_t workerCount();

}  // namespace plegl
