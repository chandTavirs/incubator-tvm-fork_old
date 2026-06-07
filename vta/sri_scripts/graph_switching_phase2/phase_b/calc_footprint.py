"""Compute derived-weight DRAM footprint for K pre-derived OFA subnets.

Reports naive (K x per-subnet) vs deduplicated (identical derivations stored once)
int8 footprint, to decide if pre-derive-all-and-cache fits device DDR.

A derived conv kernel is uniquely identified by:
  (base_weight_key, out_start:out_end, in_start:in_end, active_ks, transform_sequence)
Two subnets sharing that tuple share the exact derived bytes.
"""
import os, sys, json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TVM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
for p in ["/home/srchand/Desktop/research/OFA_Obfs",
          os.path.join(TVM_ROOT, "python"), os.path.join(TVM_ROOT, "vta", "python"),
          SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from ofa_base_models import OFADynamicResnetAllMod
from ofa_derivation_extractor import OFADerivationExtractor

OFA_CKPT = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
ARCH_FILE = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
SA_FILE = "/home/srchand/Desktop/research/OFA_Obfs/optimization_experiments/simulated_annealing/results/sa_results_20260216-175221.json"
INPUT_SHAPE = [1, 3, 224, 224]


def load_arch_mapping(f):
    data = json.load(open(f))
    if isinstance(data, dict) and "architectures" in data:
        return {item["id"]: item["architecture"] for item in data["architectures"]}
    return data


def sa_ids(n=25, lam=4.0, seed=0):
    res = json.load(open(SA_FILE))
    for r in res.get("runs", []):
        if r.get("N") == n and abs(float(r.get("lambda", -1)) - lam) < 1e-9 and r.get("seed") == seed:
            return r.get("ids", [])
    return []


def derivs_for(extractor, arch):
    """Return list of (dedup_key, nbytes) for each derived conv kernel in the subnet."""
    ds = extractor.extract_subnet_derivations(arch, INPUT_SHAPE)
    out = []
    for d in ds:
        ks = int(d.active_kernel_size)
        nbytes = int(d.out_ch) * int(d.in_ch) * ks * ks  # int8 = 1 byte/elem
        key = (d.base_weight_key, int(d.out_start), int(d.out_end),
               int(d.in_start), int(d.in_end), ks, tuple(d.transform_sequence))
        out.append((key, nbytes))
    return out


def report(name, arch_list, extractor):
    per_subnet = []
    naive = 0
    uniq = {}  # key -> nbytes
    for arch in arch_list:
        dv = derivs_for(extractor, arch)
        sz = sum(nb for _, nb in dv)
        per_subnet.append(sz)
        naive += sz
        for k, nb in dv:
            uniq[k] = nb
    dedup = sum(uniq.values())
    MB = 1024.0 * 1024.0
    K = len(arch_list)
    print("  %-22s K=%-4d  per-subnet avg=%.1f MB  naive=%.1f MB  dedup=%.1f MB  (%.1fx saving, %d uniq kernels)"
          % (name, K, (naive / K) / MB, naive / MB, dedup / MB, (naive / dedup if dedup else 0), len(uniq)))
    return naive / MB, dedup / MB


def main():
    print("Loading OFA model + arch set ...")
    net = OFADynamicResnetAllMod()
    ck = torch.load(OFA_CKPT, map_location="cpu")
    net.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    net.eval()
    ext = OFADerivationExtractor(net, verbose=False)

    arch_map = load_arch_mapping(ARCH_FILE)
    all_ids = list(arch_map.keys())
    print("  candidate architectures available: %d" % len(all_ids))

    # K=25: the actual SA subset (n=25, lambda=4, seed=0)
    sa = [i for i in sa_ids(25, 4.0, 0) if i in arch_map]
    print("\nReal SA subset (n=25, lambda=4.0, seed=0): %d ids" % len(sa))
    if sa:
        report("SA-subset", [arch_map[i] for i in sa], ext)

    print("\nSampled from candidate file (first K):")
    for K in [25, 50, 100]:
        ids = all_ids[:K]
        if len(ids) < K:
            print("  (only %d archs available, skipping K=%d)" % (len(all_ids), K))
            continue
        report("first-%d" % K, [arch_map[i] for i in ids], ext)

    print("\nDDR budget: ZCU104 = 2048 MB total; CMA probe showed >=255 MB usable for VTA buffers.")


if __name__ == "__main__":
    main()
