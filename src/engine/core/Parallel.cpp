#include "core/Parallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace plegl {
namespace {

// A minimal fork-join pool: one job at a time, workers pull chunk indices from an atomic counter.
class Pool {
 public:
  Pool() {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned i = 1; i < hw; ++i) threads_.emplace_back([this] { workerLoop(); });
  }

  ~Pool() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
  }

  std::size_t size() const { return threads_.size() + 1; }

  void run(std::size_t chunks, const std::function<void(std::size_t)>& chunkFn) {
    std::lock_guard jobLock(jobMutex_);  // Nested or concurrent calls serialize here.
    {
      std::lock_guard lock(mutex_);
      job_ = &chunkFn;
      chunkCount_ = chunks;
      nextChunk_.store(0, std::memory_order_relaxed);
      pending_ = threads_.size();
      ++generation_;
    }
    wake_.notify_all();
    drain();
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
    job_ = nullptr;
  }

 private:
  void drain() {
    for (;;) {
      const std::size_t c = nextChunk_.fetch_add(1, std::memory_order_relaxed);
      if (c >= chunkCount_) break;
      (*job_)(c);
    }
  }

  void workerLoop() {
    std::uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] { return generation_ != seen; });
        seen = generation_;
        if (stop_) return;
      }
      drain();
      {
        std::lock_guard lock(mutex_);
        if (--pending_ == 0) done_.notify_one();
      }
    }
  }

  std::vector<std::thread> threads_;
  std::mutex jobMutex_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  const std::function<void(std::size_t)>* job_ = nullptr;
  std::size_t chunkCount_ = 0;
  std::atomic<std::size_t> nextChunk_{0};
  std::size_t pending_ = 0;
  std::uint64_t generation_ = 0;
  bool stop_ = false;
};

Pool& pool() {
  static Pool p;
  return p;
}

thread_local bool tInsideParallel = false;

}  // namespace

std::size_t workerCount() { return pool().size(); }

void parallelFor(std::size_t begin, std::size_t end, std::size_t grain,
                 const std::function<void(std::size_t, std::size_t)>& fn) {
  if (end <= begin) return;
  const std::size_t n = end - begin;
  grain = std::max<std::size_t>(grain, 1);
  // Run inline when the range is small or we are already inside a parallel region.
  if (n <= grain || tInsideParallel || pool().size() == 1) {
    fn(begin, end);
    return;
  }
  const std::size_t maxChunks = pool().size() * 4;
  const std::size_t chunks = std::min(maxChunks, (n + grain - 1) / grain);
  const std::size_t chunkSize = (n + chunks - 1) / chunks;
  pool().run(chunks, [&](std::size_t c) {
    const std::size_t b = begin + c * chunkSize;
    const std::size_t e = std::min(end, b + chunkSize);
    if (b >= e) return;
    tInsideParallel = true;
    fn(b, e);
    tInsideParallel = false;
  });
}

}  // namespace plegl
