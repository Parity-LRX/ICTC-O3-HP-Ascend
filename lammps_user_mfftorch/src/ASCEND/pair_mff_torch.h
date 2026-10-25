#ifdef PAIR_CLASS
PairStyle(mff/torch,PairMFFTorch);
#else
#ifndef LMP_ASCEND_PAIR_MFF_TORCH_H
#define LMP_ASCEND_PAIR_MFF_TORCH_H

#include "pair.h"
#include <cstdint>
#include <string>
#include <vector>

struct _object;
namespace LAMMPS_NS {
class PairMFFTorch : public Pair {
 public:
  explicit PairMFFTorch(class LAMMPS *);
  ~PairMFFTorch() override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void compute(int, int) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double memory_usage() override;

 private:
  void allocate();
  void load_engine();
  void update_topology();
  void exchange_plan();
  void update_positions();
  double cutoff_ = 0;
  int depth_ = 0;
  int halo_depth_ = 0;
  int requested_depth_ = 0;
  int local_device_ = 0;
  bigint neighbor_generation_ = -1;
  int nodes_ = 0;
  double topology_seconds_ = 0, positions_seconds_ = 0, output_seconds_ = 0;
  std::string checkpoint_;
  std::vector<int64_t> type_to_z_;
  std::vector<int64_t> owner_rank_, owner_index_;
  std::vector<int> model_to_atom_;
  std::vector<int64_t> numbers_, source_, destination_, ghosts_, slots_, receive_indices_;
  std::vector<int64_t> send_counts_, receive_counts_;
  std::vector<float> positions_, shifts_, force_result_, energy_result_;
  float cell_[9] = {};
  _object *engine_ = nullptr;
};
}
#endif
#endif
