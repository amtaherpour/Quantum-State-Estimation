# regions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

from config import ExperimentConfig, RegionConfig
from core_ops import subsystem_dimensions_from_qubits


# ============================================================
# Small helpers
# ============================================================

def _ensure_nonnegative_int(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


# ============================================================
# Data containers
# ============================================================

@dataclass(frozen=True)
class RegionInfo:
    """
    Derived structural information for one region.

    Attributes
    ----------
    index :
        Region index in cfg.regions.
    name :
        Region name.
    sites :
        Global site indices belonging to the region.
    qubits :
        Total number of qubits in the region.
    dim :
        Hilbert-space dimension of the region.
    site_dims :
        Site Hilbert-space dimensions inside the region.
    shots :
        Number of measurement shots assigned to the region.
    povm_type :
        POVM family name for the region.
    povm_num_outcomes :
        Requested number of POVM outcomes, if specified.
    neighbors :
        Tuple of neighboring region indices that overlap with this region.
    """
    index: int
    name: str
    sites: Tuple[int, ...]
    qubits: int
    dim: int
    site_dims: Tuple[int, ...]
    shots: int
    povm_type: str
    povm_num_outcomes: int | None
    neighbors: Tuple[int, ...]


@dataclass(frozen=True)
class OverlapInfo:
    """
    Structural information for one overlapping region pair.

    All ordering conventions are canonical:
    the overlap sites are listed in increasing global site index order.

    Attributes
    ----------
    pair :
        Region-index pair (i, j) with i < j.
    region_names :
        Corresponding region names.
    overlap_sites :
        Shared global site indices, in increasing order.
    overlap_qubits :
        Total number of qubits in the overlap.
    overlap_dim :
        Hilbert-space dimension of the overlap subsystem.
    overlap_site_dims :
        Site Hilbert-space dimensions of the overlap subsystem.
    local_keep_i :
        Local subsystem indices inside region i that correspond to overlap_sites.
    local_keep_j :
        Local subsystem indices inside region j that correspond to overlap_sites.
    """
    pair: Tuple[int, int]
    region_names: Tuple[str, str]
    overlap_sites: Tuple[int, ...]
    overlap_qubits: int
    overlap_dim: int
    overlap_site_dims: Tuple[int, ...]
    local_keep_i: Tuple[int, ...]
    local_keep_j: Tuple[int, ...]


# ============================================================
# Main structure object
# ============================================================

class RegionGraph:
    """
    Structural wrapper around an ExperimentConfig.

    This object centralizes all region / overlap metadata so later modules
    do not have to repeatedly recompute it.

    Main usage
    ----------
    - region metadata lookup
    - overlap metadata lookup
    - neighbor queries
    - local/global index mappings for partial traces
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

        self._region_infos: Tuple[RegionInfo, ...] = self._build_region_infos()
        self._region_name_to_index: Dict[str, int] = {
            info.name: info.index for info in self._region_infos
        }

        self._overlap_infos: Tuple[OverlapInfo, ...] = self._build_overlap_infos()
        self._overlap_pair_to_info: Dict[Tuple[int, int], OverlapInfo] = {
            info.pair: info for info in self._overlap_infos
        }

    # --------------------------------------------------------
    # Basic properties
    # --------------------------------------------------------

    @property
    def num_regions(self) -> int:
        """Number of regions."""
        return len(self._region_infos)

    @property
    def region_infos(self) -> Tuple[RegionInfo, ...]:
        """All region metadata objects."""
        return self._region_infos

    @property
    def overlap_infos(self) -> Tuple[OverlapInfo, ...]:
        """All overlap metadata objects."""
        return self._overlap_infos

    @property
    def region_names(self) -> Tuple[str, ...]:
        """Tuple of all region names."""
        return tuple(info.name for info in self._region_infos)

    @property
    def overlap_pairs(self) -> Tuple[Tuple[int, int], ...]:
        """All overlapping region-index pairs (i, j) with i < j."""
        return tuple(info.pair for info in self._overlap_infos)

    # --------------------------------------------------------
    # Region lookup
    # --------------------------------------------------------

    def region_index(self, region: Union[str, RegionConfig, int]) -> int:
        """
        Resolve a region identifier to its integer index.
        """
        if isinstance(region, int):
            idx = _ensure_nonnegative_int(region, "region index")
            if idx >= self.num_regions:
                raise ValueError(
                    f"region index must be in [0, {self.num_regions - 1}], got {idx}."
                )
            return idx
        if isinstance(region, RegionConfig):
            name = region.name
        else:
            name = str(region)
        if name not in self._region_name_to_index:
            raise KeyError(f"No region with name '{name}' exists.")
        return self._region_name_to_index[name]

    def region_name(self, region: Union[str, RegionConfig, int]) -> str:
        """
        Resolve a region identifier to its name.
        """
        return self._region_infos[self.region_index(region)].name

    def region_info(self, region: Union[str, RegionConfig, int]) -> RegionInfo:
        """
        Return the RegionInfo object for a region.
        """
        return self._region_infos[self.region_index(region)]

    def region_sites(self, region: Union[str, RegionConfig, int]) -> Tuple[int, ...]:
        """
        Global site indices for a region.
        """
        return self.region_info(region).sites

    def region_site_dims(self, region: Union[str, RegionConfig, int]) -> Tuple[int, ...]:
        """
        Site subsystem dimensions inside a region.
        """
        return self.region_info(region).site_dims

    def region_dim(self, region: Union[str, RegionConfig, int]) -> int:
        """
        Hilbert-space dimension of a region.
        """
        return self.region_info(region).dim

    def region_qubits(self, region: Union[str, RegionConfig, int]) -> int:
        """
        Total number of qubits in a region.
        """
        return self.region_info(region).qubits

    def neighbors(self, region: Union[str, RegionConfig, int]) -> Tuple[int, ...]:
        """
        Neighboring region indices for a region.
        """
        return self.region_info(region).neighbors

    def neighbor_names(self, region: Union[str, RegionConfig, int]) -> Tuple[str, ...]:
        """
        Neighboring region names for a region.
        """
        return tuple(self._region_infos[j].name for j in self.neighbors(region))

    # --------------------------------------------------------
    # Local/global index maps
    # --------------------------------------------------------

    def global_to_local_site_index(
        self,
        region: Union[str, RegionConfig, int],
        global_site: int,
    ) -> int:
        """
        Map a global site index to the local subsystem index within a region.
        """
        global_site = int(global_site)
        sites = self.region_sites(region)
        if global_site not in sites:
            raise ValueError(
                f"Global site {global_site} is not contained in region '{self.region_name(region)}'."
            )
        return sites.index(global_site)

    def global_sites_to_local_keep_indices(
        self,
        region: Union[str, RegionConfig, int],
        global_sites: Sequence[int],
    ) -> Tuple[int, ...]:
        """
        Convert an ordered list of global sites into local subsystem indices
        inside the specified region.

        The output order matches the input `global_sites` order.
        """
        return tuple(
            self.global_to_local_site_index(region, s)
            for s in global_sites
        )

    # --------------------------------------------------------
    # Overlap lookup
    # --------------------------------------------------------

    def has_overlap(
        self,
        region_i: Union[str, RegionConfig, int],
        region_j: Union[str, RegionConfig, int],
    ) -> bool:
        """
        Check whether two regions overlap.
        """
        i = self.region_index(region_i)
        j = self.region_index(region_j)
        if i == j:
            return False
        pair = (min(i, j), max(i, j))
        return pair in self._overlap_pair_to_info

    def overlap_info(
        self,
        region_i: Union[str, RegionConfig, int],
        region_j: Union[str, RegionConfig, int],
    ) -> OverlapInfo:
        """
        Return OverlapInfo for a pair of overlapping regions.
        """
        i = self.region_index(region_i)
        j = self.region_index(region_j)
        if i == j:
            raise ValueError("A region does not define a nontrivial overlap with itself.")
        pair = (min(i, j), max(i, j))
        if pair not in self._overlap_pair_to_info:
            raise ValueError(
                f"Regions '{self.region_name(i)}' and '{self.region_name(j)}' do not overlap."
            )
        return self._overlap_pair_to_info[pair]

    def overlap_sites(
        self,
        region_i: Union[str, RegionConfig, int],
        region_j: Union[str, RegionConfig, int],
    ) -> Tuple[int, ...]:
        """
        Shared global site indices for an overlapping pair.
        """
        return self.overlap_info(region_i, region_j).overlap_sites

    def overlap_site_dims(
        self,
        region_i: Union[str, RegionConfig, int],
        region_j: Union[str, RegionConfig, int],
    ) -> Tuple[int, ...]:
        """
        Site subsystem dimensions on the overlap.
        """
        return self.overlap_info(region_i, region_j).overlap_site_dims

    def overlap_qubits(
        self,
        region_i: Union[str, RegionConfig, int],
        region_j: Union[str, RegionConfig, int],
    ) -> int:
        """
        Number of qubits in the overlap.
        """
        return self.overlap_info(region_i, region_j).overlap_qubits

    def overlap_dim(
        self,
        region_i: Union[str, RegionConfig, int],
        region_j: Union[str, RegionConfig, int],
    ) -> int:
        """
        Hilbert-space dimension of the overlap subsystem.
        """
        return self.overlap_info(region_i, region_j).overlap_dim

    def overlap_local_keep_indices(
        self,
        region_i: Union[str, RegionConfig, int],
        region_j: Union[str, RegionConfig, int],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Return local keep-indices for the overlap inside each region.

        Returns
        -------
        tuple
            (keep_in_region_i, keep_in_region_j)

        Both keep-index tuples correspond to the same canonical ordering of
        overlap sites: increasing global site index.
        """
        info = self.overlap_info(region_i, region_j)
        i = self.region_index(region_i)
        j = self.region_index(region_j)

        if info.pair == (i, j):
            return info.local_keep_i, info.local_keep_j
        return info.local_keep_j, info.local_keep_i

    # --------------------------------------------------------
    # Consensus variable keys
    # --------------------------------------------------------

    def canonical_overlap_key(
        self,
        region_i: Union[str, RegionConfig, int],
        region_j: Union[str, RegionConfig, int],
    ) -> Tuple[str, str]:
        """
        Canonical string key for an overlap consensus variable.

        Returns
        -------
        tuple[str, str]
            (name_small_index, name_large_index)
        """
        i = self.region_index(region_i)
        j = self.region_index(region_j)
        if i == j:
            raise ValueError("Overlap key requires two distinct regions.")
        a, b = min(i, j), max(i, j)
        return self._region_infos[a].name, self._region_infos[b].name

    def directed_dual_key(
        self,
        source_region: Union[str, RegionConfig, int],
        target_region: Union[str, RegionConfig, int],
    ) -> Tuple[str, str]:
        """
        Directed key for dual variables Lambda_{source,target}.
        """
        i = self.region_index(source_region)
        j = self.region_index(target_region)
        if i == j:
            raise ValueError("Dual key requires two distinct regions.")
        if not self.has_overlap(i, j):
            raise ValueError(
                f"Regions '{self.region_name(i)}' and '{self.region_name(j)}' do not overlap."
            )
        return self.region_name(i), self.region_name(j)

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    def region_summary_dicts(self) -> List[Dict[str, object]]:
        """
        Return region metadata as a list of dictionaries.
        """
        out: List[Dict[str, object]] = []
        for info in self._region_infos:
            out.append(
                {
                    "index": info.index,
                    "name": info.name,
                    "sites": info.sites,
                    "qubits": info.qubits,
                    "dim": info.dim,
                    "site_dims": info.site_dims,
                    "shots": info.shots,
                    "povm_type": info.povm_type,
                    "povm_num_outcomes": info.povm_num_outcomes,
                    "neighbors": info.neighbors,
                }
            )
        return out

    def overlap_summary_dicts(self) -> List[Dict[str, object]]:
        """
        Return overlap metadata as a list of dictionaries.
        """
        out: List[Dict[str, object]] = []
        for info in self._overlap_infos:
            out.append(
                {
                    "pair": info.pair,
                    "region_names": info.region_names,
                    "overlap_sites": info.overlap_sites,
                    "overlap_qubits": info.overlap_qubits,
                    "overlap_dim": info.overlap_dim,
                    "overlap_site_dims": info.overlap_site_dims,
                    "local_keep_i": info.local_keep_i,
                    "local_keep_j": info.local_keep_j,
                }
            )
        return out

    def pretty_print(self) -> None:
        """
        Print a readable summary of regions and overlaps.
        """
        print("=" * 72)
        print("RegionGraph summary")
        print("-" * 72)
        print(f"Number of regions: {self.num_regions}")
        print(f"Region names: {self.region_names}")
        print(f"Overlap pairs: {self.overlap_pairs}")
        print("-" * 72)
        print("Regions")
        for info in self._region_infos:
            print(
                f"[{info.index}] name={info.name}, sites={info.sites}, "
                f"q={info.qubits}, dim={info.dim}, site_dims={info.site_dims}, "
                f"neighbors={info.neighbors}"
            )
        print("-" * 72)
        print("Overlaps")
        if len(self._overlap_infos) == 0:
            print("(none)")
        else:
            for info in self._overlap_infos:
                print(
                    f"{info.region_names[0]} <-> {info.region_names[1]} : "
                    f"sites={info.overlap_sites}, q={info.overlap_qubits}, "
                    f"dim={info.overlap_dim}, keep_i={info.local_keep_i}, "
                    f"keep_j={info.local_keep_j}"
                )
        print("=" * 72)

    # --------------------------------------------------------
    # Internal builders
    # --------------------------------------------------------

    def _build_region_infos(self) -> Tuple[RegionInfo, ...]:
        infos: List[RegionInfo] = []
        for idx, region in enumerate(self.cfg.regions):
            neighbors = self.cfg.neighbors(idx)
            info = RegionInfo(
                index=idx,
                name=region.name,
                sites=tuple(region.sites),
                qubits=self.cfg.region_qubits(region),
                dim=self.cfg.region_dimension(region),
                site_dims=self.cfg.region_site_dimensions(region),
                shots=region.shots,
                povm_type=region.povm_type,
                povm_num_outcomes=region.povm_num_outcomes,
                neighbors=tuple(neighbors),
            )
            infos.append(info)
        return tuple(infos)

    def _build_overlap_infos(self) -> Tuple[OverlapInfo, ...]:
        infos: List[OverlapInfo] = []
        site_dims_global = subsystem_dimensions_from_qubits(self.cfg.qubits_per_site)

        for i, j in self.cfg.overlap_pairs():
            region_i = self.cfg.regions[i]
            region_j = self.cfg.regions[j]

            overlap_sites = tuple(sorted(set(region_i.sites).intersection(region_j.sites)))
            overlap_qubits = int(sum(self.cfg.qubits_per_site[s] for s in overlap_sites))
            overlap_site_dims = tuple(site_dims_global[s] for s in overlap_sites)

            overlap_dim = 1
            for d in overlap_site_dims:
                overlap_dim *= d

            local_keep_i = tuple(region_i.sites.index(s) for s in overlap_sites)
            local_keep_j = tuple(region_j.sites.index(s) for s in overlap_sites)

            info = OverlapInfo(
                pair=(i, j),
                region_names=(region_i.name, region_j.name),
                overlap_sites=overlap_sites,
                overlap_qubits=overlap_qubits,
                overlap_dim=overlap_dim,
                overlap_site_dims=overlap_site_dims,
                local_keep_i=local_keep_i,
                local_keep_j=local_keep_j,
            )
            infos.append(info)

        return tuple(infos)


# ============================================================
# Convenience free functions
# ============================================================

def build_region_graph(cfg: ExperimentConfig) -> RegionGraph:
    """
    Convenience constructor for RegionGraph.
    """
    return RegionGraph(cfg)


def region_name_to_index_map(cfg: ExperimentConfig) -> Dict[str, int]:
    """
    Return a simple region-name to region-index mapping.
    """
    return {region.name: idx for idx, region in enumerate(cfg.regions)}


def overlap_name_pairs(cfg: ExperimentConfig) -> Tuple[Tuple[str, str], ...]:
    """
    Return overlapping region pairs by name.
    """
    out: List[Tuple[str, str]] = []
    for i, j in cfg.overlap_pairs():
        out.append((cfg.regions[i].name, cfg.regions[j].name))
    return tuple(out)


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_default_graph() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    graph = RegionGraph(cfg)

    assert graph.num_regions == 2
    assert graph.region_names == ("R0", "R1")
    assert graph.overlap_pairs == ((0, 1),)

    info0 = graph.region_info("R0")
    info1 = graph.region_info("R1")

    assert info0.sites == (0, 1)
    assert info1.sites == (1, 2)
    assert info0.dim == 4
    assert info1.dim == 4
    assert graph.neighbor_names("R0") == ("R1",)
    assert graph.neighbor_names("R1") == ("R0",)

    ov = graph.overlap_info("R0", "R1")
    assert ov.overlap_sites == (1,)
    assert ov.overlap_qubits == 1
    assert ov.overlap_dim == 2
    assert ov.local_keep_i == (1,)
    assert ov.local_keep_j == (0,)


def _self_test_local_index_maps() -> None:
    from config import ExperimentConfig, RegionConfig

    # Region A: sites (0,1,3) -> qubits = 1+2+1 = 4 -> dim = 16
    # Region B: sites (1,2,3) -> qubits = 2+1+1 = 4 -> dim = 16
    # For random_ic, config.py requires povm_num_outcomes >= dim^2 = 256.
    cfg = ExperimentConfig(
        qubits_per_site=(1, 2, 1, 1),
        regions=(
            RegionConfig(
                name="A",
                sites=(0, 1, 3),
                shots=100,
                povm_type="computational",
                povm_num_outcomes=16,
            ),
            RegionConfig(
                name="B",
                sites=(1, 2, 3),
                shots=100,
                povm_type="random_ic",
                povm_num_outcomes=256,
            ),
        ),
        experiment_name="index_map_test",
    )

    graph = RegionGraph(cfg)

    assert graph.global_to_local_site_index("A", 0) == 0
    assert graph.global_to_local_site_index("A", 1) == 1
    assert graph.global_to_local_site_index("A", 3) == 2

    assert graph.global_sites_to_local_keep_indices("A", (1, 3)) == (1, 2)
    assert graph.global_sites_to_local_keep_indices("B", (1, 3)) == (0, 2)

    ov = graph.overlap_info("A", "B")
    assert ov.overlap_sites == (1, 3)
    assert ov.local_keep_i == (1, 2)
    assert ov.local_keep_j == (0, 2)


def _self_test_overlap_keys() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    graph = RegionGraph(cfg)

    assert graph.canonical_overlap_key("R0", "R1") == ("R0", "R1")
    assert graph.canonical_overlap_key("R1", "R0") == ("R0", "R1")

    assert graph.directed_dual_key("R0", "R1") == ("R0", "R1")
    assert graph.directed_dual_key("R1", "R0") == ("R1", "R0")


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the regions module.
    """
    tests = [
        ("default region graph", _self_test_default_graph),
        ("local/global index maps", _self_test_local_index_maps),
        ("overlap and dual keys", _self_test_overlap_keys),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All regions self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)

    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    graph = RegionGraph(cfg)
    graph.pretty_print()
