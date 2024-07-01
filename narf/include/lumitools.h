#include <unordered_map>
#include <memory>
#include <algorithm>
#include <stdexcept>

struct RunLumiHash {
  std::size_t operator()(const std::pair<unsigned int, unsigned int> &pair) const {
    return std::hash<unsigned long long>{}(((unsigned long long)(pair.first) << 32) + (unsigned long long)(pair.second));
  }
};

class BrilcalcHelper {
 
public:
  using valuemap_t = std::unordered_map<std::pair<unsigned int, unsigned int>, double, RunLumiHash>;
  
  BrilcalcHelper(const std::vector<unsigned int> &runs, const std::vector<unsigned int> &lumis, const std::vector<double> &lumivals) :
  valuemap_(std::make_shared<valuemap_t>()) {
    for (unsigned int i = 0; i < lumivals.size(); ++i) {
      valuemap_->insert(std::make_pair(std::make_pair(runs[i], lumis[i]), lumivals[i]));
    }
  }
    
  double operator () (unsigned int run, unsigned int lumi) const {
    const auto it = valuemap_->find(std::make_pair(run, lumi));
    if (it != valuemap_->end()) {
      return it->second;
    }
    throw std::runtime_error("lumi not found");
    
    return 0.;
  }
  
  
private:
  std::shared_ptr<valuemap_t> valuemap_;
  
};

class JsonHelper {
public:
  using pair_t = std::pair<unsigned int, unsigned int>;
  using jsonmap_t = std::unordered_map<unsigned int, std::vector<pair_t>>;
  
  JsonHelper(const std::vector<unsigned int> &runs, const std::vector<unsigned int> &firstlumis, const std::vector<unsigned int> &lastlumis) :
  jsonmap_(std::make_shared<jsonmap_t>()) {
    for (unsigned int i = 0; i < firstlumis.size(); ++i) {
      (*jsonmap_)[runs[i]].push_back(std::make_pair(firstlumis[i],lastlumis[i]));
    }
    
    for (auto &item : *jsonmap_) {
      std::sort(item.second.begin(), item.second.end());
    }
  }
  
  bool operator () (unsigned int run, unsigned int lumi) const {
    if (run == 1) {
      return true;
    }
    
    const auto it = jsonmap_->find(run);
    if (it != jsonmap_->end()) {
      auto const &pairs = it->second;
      auto const pairit = std::lower_bound(pairs.begin(), pairs.end(), lumi, [](const pair_t &pair, unsigned int val) { return pair.second < val; } );
      if (pairit != pairs.end()) {
        if (lumi >= pairit->first) {
          return true;
        }
      }
    }
    return false;
  }
  

private:
  std::shared_ptr<jsonmap_t> jsonmap_;
};
