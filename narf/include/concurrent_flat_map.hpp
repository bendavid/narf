#ifndef NARF_CONCURRENT_FLAT_MAP_HPP
#define NARF_CONCURRENT_FLAT_MAP_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace narf {

// Lock-free, insert-only, expandable concurrent flat hash map.
//
// Properties:
//   - find / insert / emplace / expansion are all lock-free and safe to call
//     concurrently from any number of threads.
//   - Elements are never erased; once inserted, the address of an element's
//     value is stable for the lifetime of the map (pointers returned by
//     find/emplace remain valid).
//   - The container grows by appending geometrically larger segments; existing
//     segments are never rehashed, so concurrent readers are never disturbed.
//
// Requirements on Key:
//   - Must be an integer type.
//   - The two most significant bits of any user-supplied key must be zero;
//     they are reserved for internal slot state (occupied / busy markers).
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class concurrent_flat_map {
  static_assert(std::is_integral_v<Key>, "Key must be an integer type");

public:
  using key_type    = Key;
  using mapped_type = Value;
  using hasher      = Hash;

private:
  using UKey = std::make_unsigned_t<Key>;

  static constexpr unsigned kKeyBits   = sizeof(UKey) * 8;
  static constexpr UKey kOccupiedBit   = UKey(1) << (kKeyBits - 1);
  static constexpr UKey kBusyBit       = UKey(1) << (kKeyBits - 2);
  static constexpr UKey kStateMask     = kOccupiedBit | kBusyBit;
  static constexpr UKey kPayloadMask   = ~kStateMask;
  static constexpr UKey kEmpty         = 0;

  struct Slot {
    std::atomic<UKey> key{kEmpty};
    alignas(Value) unsigned char storage[sizeof(Value)];

    Value* value_ptr() noexcept {
      return std::launder(reinterpret_cast<Value*>(&storage));
    }
  };

  struct Segment {
    const std::size_t capacity;
    const std::size_t mask;
    std::atomic<std::size_t> size{0};
    std::unique_ptr<Slot[]> slots;
    std::atomic<Segment*> next{nullptr};

    explicit Segment(std::size_t cap)
      : capacity(cap), mask(cap - 1), slots(new Slot[cap]) {}

    ~Segment() {
      if constexpr (!std::is_trivially_destructible_v<Value>) {
        for (std::size_t i = 0; i < capacity; ++i) {
          UKey k = slots[i].key.load(std::memory_order_relaxed);
          if ((k & kOccupiedBit) && !(k & kBusyBit)) {
            slots[i].value_ptr()->~Value();
          }
        }
      }
    }
  };

  static constexpr std::size_t kDefaultInitialCapacity = 64;
  static constexpr std::size_t kMaxProbe               = 32;

  Segment* head_;
  std::atomic<Segment*> tail_;
  Hash hash_;

  static UKey encode(Key key) noexcept {
    return (static_cast<UKey>(key) & kPayloadMask) | kOccupiedBit;
  }

  static std::size_t round_up_pow2(std::size_t n) noexcept {
    std::size_t c = 1;
    while (c < n) c <<= 1;
    return c;
  }

  // Spin until the slot's busy bit clears, then return the stable key.
  static UKey wait_not_busy(Slot& slot) noexcept {
    UKey k = slot.key.load(std::memory_order_acquire);
    while (k & kBusyBit) {
      k = slot.key.load(std::memory_order_acquire);
    }
    return k;
  }

  // Allocate (or observe) the segment that follows `current`. Multiple threads
  // racing here will agree on a single winning segment; losers free their
  // speculative allocation.
  Segment* ensure_next(Segment* current) {
    Segment* next = current->next.load(std::memory_order_acquire);
    if (next) return next;
    auto* fresh = new Segment(current->capacity * 2);
    Segment* expected = nullptr;
    if (current->next.compare_exchange_strong(
            expected, fresh,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      // Best-effort tail advance so future inserters skip filled segments.
      Segment* t = tail_.load(std::memory_order_acquire);
      while (true) {
        Segment* tn = t->next.load(std::memory_order_acquire);
        if (!tn) break;
        if (tail_.compare_exchange_weak(t, tn,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
          t = tn;
        }
      }
      return fresh;
    }
    delete fresh;
    return expected;
  }

  // Search a single segment for `target`. Returns pointer to value or nullptr.
  Value* find_in(Segment* seg, std::size_t h, UKey target) const noexcept {
    const std::size_t base = h & seg->mask;
    const std::size_t probe_limit = std::min(kMaxProbe, seg->capacity);
    for (std::size_t i = 0; i < probe_limit; ++i) {
      Slot& slot = seg->slots[(base + i) & seg->mask];
      UKey k = slot.key.load(std::memory_order_acquire);
      if (k == kEmpty) return nullptr;
      if (k & kBusyBit) k = wait_not_busy(slot);
      if (k == target) return slot.value_ptr();
    }
    return nullptr;
  }

  // Try to insert `target` into a single segment. Returns
  //   {ptr, true}  : newly inserted, value constructed from args
  //   {ptr, false} : key was already present
  //   {nullptr, false} : segment full along this probe sequence
  template <typename... Args>
  std::pair<Value*, bool> emplace_in(Segment* seg, std::size_t h, UKey target,
                                     Args&&... args) {
    const std::size_t base = h & seg->mask;
    const std::size_t probe_limit = std::min(kMaxProbe, seg->capacity);
    const UKey busy = (target & kPayloadMask) | kOccupiedBit | kBusyBit;
    for (std::size_t i = 0; i < probe_limit; ++i) {
      Slot& slot = seg->slots[(base + i) & seg->mask];
      UKey k = slot.key.load(std::memory_order_acquire);
      if (k == kEmpty) {
        UKey expected = kEmpty;
        if (slot.key.compare_exchange_strong(
                expected, busy,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
          ::new (static_cast<void*>(&slot.storage))
              Value(std::forward<Args>(args)...);
          slot.key.store(target, std::memory_order_release);
          seg->size.fetch_add(1, std::memory_order_relaxed);
          return {slot.value_ptr(), true};
        }
        k = expected;
      }
      if (k & kBusyBit) k = wait_not_busy(slot);
      if (k == target) return {slot.value_ptr(), false};
    }
    return {nullptr, false};
  }

public:
  explicit concurrent_flat_map(std::size_t initial_capacity = kDefaultInitialCapacity) {
    const std::size_t cap = round_up_pow2(std::max<std::size_t>(initial_capacity, 2));
    head_ = new Segment(cap);
    tail_.store(head_, std::memory_order_release);
  }

  ~concurrent_flat_map() {
    Segment* s = head_;
    while (s) {
      Segment* n = s->next.load(std::memory_order_relaxed);
      delete s;
      s = n;
    }
  }

  concurrent_flat_map(const concurrent_flat_map&)            = delete;
  concurrent_flat_map& operator=(const concurrent_flat_map&) = delete;

  // Move constructor: transfers ownership. The moved-from object is left in
  // a destroy-only state; calling any other method on it is undefined.
  concurrent_flat_map(concurrent_flat_map&& other) noexcept
    : head_(other.head_),
      tail_(other.tail_.load(std::memory_order_relaxed)),
      hash_(std::move(other.hash_)) {
    other.head_ = nullptr;
    other.tail_.store(nullptr, std::memory_order_relaxed);
  }

  concurrent_flat_map& operator=(concurrent_flat_map&& other) noexcept {
    if (this != &other) {
      Segment* s = head_;
      while (s) {
        Segment* n = s->next.load(std::memory_order_relaxed);
        delete s;
        s = n;
      }
      head_ = other.head_;
      tail_.store(other.tail_.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
      hash_ = std::move(other.hash_);
      other.head_ = nullptr;
      other.tail_.store(nullptr, std::memory_order_relaxed);
    }
    return *this;
  }

  // Returns a pointer to the value associated with `key`, or nullptr.
  // The returned pointer remains valid for the lifetime of the map.
  Value* find(Key key) noexcept {
    const UKey target  = encode(key);
    const std::size_t h = hash_(key);
    for (Segment* seg = head_; seg;
         seg = seg->next.load(std::memory_order_acquire)) {
      if (Value* v = find_in(seg, h, target)) return v;
    }
    return nullptr;
  }

  const Value* find(Key key) const noexcept {
    return const_cast<concurrent_flat_map*>(this)->find(key);
  }

  bool contains(Key key) const noexcept { return find(key) != nullptr; }

  // Construct a value in-place if `key` is not yet present.
  // Returns {pointer-to-value, inserted?}.
  template <typename... Args>
  std::pair<Value*, bool> emplace(Key key, Args&&... args) {
    const UKey target  = encode(key);
    const std::size_t h = hash_(key);

    // Look in every existing segment first to honour insert-once semantics.
    for (Segment* seg = head_; seg;
         seg = seg->next.load(std::memory_order_acquire)) {
      if (Value* v = find_in(seg, h, target)) return {v, false};
    }

    // Then try to insert at the tail, growing as required.
    Segment* seg = tail_.load(std::memory_order_acquire);
    while (true) {
      auto result = emplace_in(seg, h, target, std::forward<Args>(args)...);
      if (result.first) {
        return result;
      }
      // This probe sequence is saturated in `seg`; another thread may have
      // already inserted the same key into a later segment, so re-check
      // everything past `seg` before allocating.
      Segment* next = seg->next.load(std::memory_order_acquire);
      if (!next) next = ensure_next(seg);
      for (Segment* s = next; s; s = s->next.load(std::memory_order_acquire)) {
        if (Value* v = find_in(s, h, target)) return {v, false};
      }
      seg = next;
    }
  }

  std::pair<Value*, bool> insert(Key key, const Value& v) {
    return emplace(key, v);
  }
  std::pair<Value*, bool> insert(Key key, Value&& v) {
    return emplace(key, std::move(v));
  }

  // Total number of inserted elements summed across all segments. Reads each
  // segment's atomic counter; safe to call concurrently with other operations
  // but the result reflects an instantaneous, possibly racing snapshot.
  std::size_t size() const noexcept {
    std::size_t total = 0;
    for (Segment* seg = head_; seg;
         seg = seg->next.load(std::memory_order_acquire)) {
      total += seg->size.load(std::memory_order_acquire);
    }
    return total;
  }

  // Visit every (key, value) pair currently in the map. Not safe to call
  // concurrently with insertions if the visitor relies on a stable snapshot;
  // the visitor only sees fully-constructed slots and waits past any in-flight
  // insert that races with iteration.
  template <typename F>
  void for_each(F&& f) {
    for (Segment* seg = head_; seg;
         seg = seg->next.load(std::memory_order_acquire)) {
      for (std::size_t i = 0; i < seg->capacity; ++i) {
        Slot& slot = seg->slots[i];
        UKey k = slot.key.load(std::memory_order_acquire);
        if (k == kEmpty) continue;
        if (k & kBusyBit) k = wait_not_busy(slot);
        if (!(k & kOccupiedBit)) continue;
        f(static_cast<Key>(k & kPayloadMask), *slot.value_ptr());
      }
    }
  }

  template <typename F>
  void for_each(F&& f) const {
    const_cast<concurrent_flat_map*>(this)->for_each(std::forward<F>(f));
  }

  // Remove all elements and shrink back to a single head segment. NOT thread
  // safe with respect to any other operation; the caller must establish
  // exclusive access (e.g. between processing passes).
  void clear() {
    Segment* s = head_->next.load(std::memory_order_relaxed);
    while (s) {
      Segment* n = s->next.load(std::memory_order_relaxed);
      delete s;
      s = n;
    }
    head_->next.store(nullptr, std::memory_order_relaxed);
    if constexpr (!std::is_trivially_destructible_v<Value>) {
      for (std::size_t i = 0; i < head_->capacity; ++i) {
        UKey k = head_->slots[i].key.load(std::memory_order_relaxed);
        if ((k & kOccupiedBit) && !(k & kBusyBit)) {
          head_->slots[i].value_ptr()->~Value();
        }
        head_->slots[i].key.store(kEmpty, std::memory_order_relaxed);
      }
    } else {
      for (std::size_t i = 0; i < head_->capacity; ++i) {
        head_->slots[i].key.store(kEmpty, std::memory_order_relaxed);
      }
    }
    head_->size.store(0, std::memory_order_relaxed);
    tail_.store(head_, std::memory_order_release);
  }
};

} // namespace narf

#endif // NARF_CONCURRENT_FLAT_MAP_HPP
