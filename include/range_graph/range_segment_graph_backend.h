/**
 * @file range_segment_graph_backend.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Construction-only segmented routing backend for RCFS v4.
 * @date 2023-06-19; Revised 2024-01-10
 *
 * @copyright Copyright (c) 2023
 */

#pragma once

#include <algorithm>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "base_hnsw/hnswalg.h"
#include "base_hnsw/hnswlib.h"

namespace rcfs_range_backend {

using namespace base_hnsw;
using std::pair;

using std::vector;

struct OneSegmentNeighbors {
  OneSegmentNeighbors() { batch = 0; }
  OneSegmentNeighbors(unsigned num) : batch(num) {}
  OneSegmentNeighbors(unsigned num, int start, int end)
      : batch(num), start(start), end(end) {}

  // vector<pair<int, float>> nns;
  vector<int> nns_id;
  unsigned batch;  // batch id
  int start = -1;  // left position
  int end = -2;    // right position
  const unsigned size() const { return nns_id.size(); }
};

// struct OneTupleNeighbor {
//   int start = -1;
//   int end = -2;
//   int nn_id;
// }

// Temporary construction output. V4 builds reverse adjacency directly into
// its flat storage, so no per-node reverse vector is needed here.
struct SegmentedForwardNeighbors {
  vector<OneSegmentNeighbors> forward_nns;
};

template <typename dist_t>
class SegmentedRangeHNSW : public HierarchicalNSW<float> {
 public:
  SegmentedRangeHNSW(SpaceInterface<float> *s, size_t max_elements,
                     size_t M, size_t ef_construction, size_t ef_max,
                     size_t random_seed)
      : HierarchicalNSW(s, max_elements, M, ef_construction, random_seed) {
    ef_max_ = ef_max;
  }

  vector<SegmentedForwardNeighbors> *segment_graph = nullptr;

  // Rewrite the searching while construting HNSW, keep more neighbors.
  // TODO: maybe can be updated to a position sensing functions.
  // or combine with rnn-descent
  virtual std::priority_queue<std::pair<dist_t, tableint>,
                              std::vector<std::pair<dist_t, tableint>>,
                              CompareByFirst>
  searchBaseLayerLevel0(tableint ep_id, const void *data_point, int layer) {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidateSet;

    std::vector<pair<dist_t, tableint>> deleted_list;

    size_t ef_construction = ef_max_;

    dist_t lowerBound;
    if (!isMarkedDeleted(ep_id)) {
      dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                 dist_func_param_);
      top_candidates.emplace(dist, ep_id);
      lowerBound = dist;
      candidateSet.emplace(-dist, ep_id);
    } else {
      lowerBound = std::numeric_limits<dist_t>::max();
      candidateSet.emplace(-lowerBound, ep_id);
    }
    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
      std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
      if ((-curr_el_pair.first) > lowerBound) {
        break;
      }
      candidateSet.pop();

      tableint curNodeNum = curr_el_pair.second;

      std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

      int *data;  // = (int *)(linkList0_ + curNodeNum *
                  // size_links_per_element0_);
      if (layer == 0) {
        data = (int *)get_linklist0(curNodeNum);
      } else {
        data = (int *)get_linklist(curNodeNum, layer);
        //                    data = (int *) (linkLists_[curNodeNum] + (layer
        //                    - 1) * size_links_per_element_);
      }
      size_t size = getListCount((linklistsizeint *)data);
      tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
        if (visited_array[candidate_id] == visited_array_tag) continue;
        visited_array[candidate_id] = visited_array_tag;
        char *currObj1 = (getDataByInternalId(candidate_id));

        dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
        if (top_candidates.size() < ef_construction || lowerBound > dist1) {
          candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
          _mm_prefetch(getDataByInternalId(candidateSet.top().second),
                       _MM_HINT_T0);
#endif

          if (!isMarkedDeleted(candidate_id))
            top_candidates.emplace(dist1, candidate_id);

          // record deleted neighbors
          if (top_candidates.size() > ef_construction) {
            deleted_list.emplace_back(top_candidates.top());
            top_candidates.pop();
          }

          if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
        }
      }
    }
    visited_list_pool_->releaseVisitedList(vl);

    // add back deleted neighbors for recursively pruning
    for (auto deleted_candidate : deleted_list) {
      top_candidates.emplace(deleted_candidate);
    }

    return top_candidates;
  }

  // rewrite connect new element, integrate recursively heuristic pruning
  virtual tableint mutuallyConnectNewElementLevel0(
      const void *data_point, tableint cur_c,
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      int level, bool isUpdate) {
    size_t Mcurmax = maxM0_;

    // forward neighbors in top candidates
    int external_id = getExternalLabel(cur_c);
    tableint next_closest_entry_point = 0;

    // original structure for fetching nodes
    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(M_);
    {
      // MAX_POS recursive pruning, combining into connect function.

      // reverse top_candidates, from cloest to farthest
      std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
      while (top_candidates.size() > 0) {
        queue_closest.emplace(-top_candidates.top().first,
                              top_candidates.top().second);
        top_candidates.pop();
      }
      // Now top_candidates is empty

      int external_left_most = -1;
      int last_batch_left_most = -1;

      unsigned iter_counter = 0;
      unsigned batch_counter = 0;
      std::vector<std::pair<dist_t, tableint>> return_list;
      std::vector<int> return_external_list;

      // Need a buffer candidates to store neighbors (external_id, dist_t,
      // internal_id)
      vector<pair<int, pair<dist_t, tableint>>> buffer_candidates;

      // using tableint_pair = std::pair<tableint, tableint>;
      // std::unordered_map<tableint_pair, dist_t, boost::hash<tableint_pair>>
      //     visited_nn_dists;

      while (queue_closest.size()) {
        // If return list size meet M or current window exceed ef_construction,
        // end this batch, enter the new batch
        if (return_list.size() >= Mcurmax ||
            iter_counter >= ef_basic_construction_) {
          // reset batch, add current batch;
          // no breaking because recursivly visiting the candidates.

          if (batch_counter == 0) {
            // The first batch, also use for original HNSW constructing
            next_closest_entry_point =
                return_list.front()
                    .second;  // TODO: check whether the nearest neighbor
            for (std::pair<dist_t, tableint> curent_pair : return_list) {
              selectedNeighbors.push_back(curent_pair.second);
            }
          }

          // TODO: For MID_POS and MIN_POS, find the external_left_most by
          // sorting.
          // const auto [min_pos, max_pos] = std::minmax_element(
          //     std::begin(return_external_list),
          //     std::end(return_external_list));
          // assert(*max_pos == external_left_most);
          // OneSegmentNeighbors one_segment(batch_counter, *min_pos, *max_pos);

          for (pair<int, pair<dist_t, tableint>> curent_buffer :
               buffer_candidates) {
            if (curent_buffer.first > external_left_most) {
              // available in next batch, add back to the queue.
              queue_closest.emplace(curent_buffer.second);
            }
          }

          // current segment start position: [last left + 1, current most]
          OneSegmentNeighbors one_segment(
              batch_counter, last_batch_left_most + 1, external_left_most);

          // for (size_t p_idx = 0; p_idx < return_list.size(); p_idx++) {
          //   // keep dists
          //   // one_segment.nns.emplace_back(return_external_list.at(p_idx),
          //   //                              -return_list.at(p_idx).first);
          // }
          // only keep id, drop dists
          one_segment.nns_id.swap(return_external_list);
          segment_graph->at(external_id).forward_nns.emplace_back(one_segment);

          return_list.clear();
          return_external_list.clear();
          iter_counter = 0;
          batch_counter++;
          last_batch_left_most = external_left_most;
        }

        // experimets show that this doesn't matter a lot
        // if queue smaller than Mcurmax, add all
        // if ((return_list.size() + queue_closest.size()) < Mcurmax) {
        //   while (queue_closest.size()) {
        //     std::pair<dist_t, tableint> curent_pair = queue_closest.top();
        //     int curent_external_id = getExternalLabel(curent_pair.second);
        //     queue_closest.pop();
        //     if (curent_external_id > external_left_most) {
        //       return_list.push_back(curent_pair);
        //       return_external_list.push_back(curent_external_id);
        //     }
        //   }
        //   break;
        // }

        std::pair<dist_t, tableint> curent_pair = queue_closest.top();
        dist_t dist_to_query = -curent_pair.first;
        queue_closest.pop();

        int curent_external_id = getExternalLabel(curent_pair.second);
        if (curent_external_id < last_batch_left_most) {
          // position left than last batch, skip
          continue;
        }
        iter_counter++;
        bool good = true;

        for (std::pair<dist_t, tableint> second_pair : return_list) {
          dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);

          // use a map structure to record the distances, maybe can boost
          // MID_POS and MIN_POS
          // dist_t curdist; tableint_pair nn_pair =
          //     std::make_pair(std::min(second_pair.second,
          //     curent_pair.second),
          //                    std::max(second_pair.second,
          //                    curent_pair.second));
          // if (visited_nn_dists.count(nn_pair)) {
          //   curdist = visited_nn_dists.at(nn_pair);
          // } else {
          //   curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
          //                          getDataByInternalId(curent_pair.second),
          //                          dist_func_param_);
          //   visited_nn_dists.insert({nn_pair, curdist});
          // }

          if (curdist < dist_to_query) {
            good = false;
            break;
          }
        }
        if (good) {
          return_list.push_back(curent_pair);
          return_external_list.push_back(curent_external_id);
          if (curent_external_id > external_left_most) {
            external_left_most = curent_external_id;
          }
        } else {
          // decide whether add to buffer list, TODO: consider MIN_POS, MID_POS
          if (curent_external_id > external_left_most) {
            buffer_candidates.emplace_back(
                make_pair(curent_external_id, curent_pair));
          }
        }
      }

      if (batch_counter == 0) {
        // The first batch, also use for original HNSW constructing
        next_closest_entry_point = return_list.front().second;
        for (std::pair<dist_t, tableint> curent_pair : return_list) {
          selectedNeighbors.push_back(curent_pair.second);
        }
      }
      if (!return_list.empty()) {
        OneSegmentNeighbors one_segment(batch_counter, last_batch_left_most + 1,
                                        external_id);
        // for (size_t p_idx = 0; p_idx < return_list.size(); p_idx++) {
        //   // keep dists
        //   // one_segment.nns.emplace_back(return_external_list.at(p_idx),
        //   //                              -return_list.at(p_idx).first);
        // }
        // only keep id, drop dists
        one_segment.nns_id.swap(return_external_list);
        segment_graph->at(external_id).forward_nns.emplace_back(one_segment);
      }
    }

    {
      linklistsizeint *ll_cur;
      ll_cur = get_linklist0(cur_c);

      if (*ll_cur && !isUpdate) {
        throw std::runtime_error(
            "The newly inserted element should have blank link list");
      }
      setListCount(ll_cur, selectedNeighbors.size());
      tableint *data = (tableint *)(ll_cur + 1);
      for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
        if (data[idx] && !isUpdate)
          throw std::runtime_error("Possible memory corruption");
        if (level > element_levels_[selectedNeighbors[idx]])
          throw std::runtime_error(
              "Trying to make a link on a non-existent level");

        data[idx] = selectedNeighbors[idx];
      }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      // Note: only keep partial m when first constructing, for removing
      // duplicates for wiki-image and yt8m-audio, just randomly pick a
      // threshold.
      // TODO: better duplicate strategy during graph contruction
      if (idx > (maxM0_ * 0.7)) {
        break;
      }
      std::unique_lock<std::mutex> lock(
          link_list_locks_[selectedNeighbors[idx]]);

      linklistsizeint *ll_other;
      ll_other = get_linklist0(selectedNeighbors[idx]);

      size_t sz_link_list_other = getListCount(ll_other);

      if (sz_link_list_other > Mcurmax)
        throw std::runtime_error("Bad value of sz_link_list_other");
      if (selectedNeighbors[idx] == cur_c)
        throw std::runtime_error("Trying to connect an element to itself");
      if (level > element_levels_[selectedNeighbors[idx]])
        throw std::runtime_error(
            "Trying to make a link on a non-existent level");

      tableint *data = (tableint *)(ll_other + 1);

      bool is_cur_c_present = false;
      if (isUpdate) {
        for (size_t j = 0; j < sz_link_list_other; j++) {
          if (data[j] == cur_c) {
            is_cur_c_present = true;
            break;
          }
        }
      }

      // If cur_c is already present in the neighboring connections of
      // `selectedNeighbors[idx]` then no need to modify any connections or
      // run the heuristics.
      if (!is_cur_c_present) {
        if (sz_link_list_other < Mcurmax) {
          data[sz_link_list_other] = cur_c;
          setListCount(ll_other, sz_link_list_other + 1);

        } else {
          // finding the "weakest" element to replace it with the new one
          dist_t d_max = fstdistfunc_(
              getDataByInternalId(cur_c),
              getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);
          // Heuristic:
          std::priority_queue<std::pair<dist_t, tableint>,
                              std::vector<std::pair<dist_t, tableint>>,
                              CompareByFirst>
              candidates;
          candidates.emplace(d_max, cur_c);

          for (size_t j = 0; j < sz_link_list_other; j++) {
            candidates.emplace(
                fstdistfunc_(getDataByInternalId(data[j]),
                             getDataByInternalId(selectedNeighbors[idx]),
                             dist_func_param_),
                data[j]);
          }

          getNeighborsByHeuristic2(candidates, Mcurmax);

          int indx = 0;
          while (candidates.size() > 0) {
            data[indx] = candidates.top().second;
            candidates.pop();
            indx++;
          }
          setListCount(ll_other, indx);
          // Nearest K:
          // int indx = -1;
          // for (int j = 0; j < sz_link_list_other; j++) {
          //     dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
          // getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) {
          //         indx = j;
          //         d_max = d;
          //     }
          // }
          // if (indx >= 0) {
          //     data[indx] = cur_c;
          // }
        }
      }
    }

    return next_closest_entry_point;
  }
};

}  // namespace rcfs_range_backend
