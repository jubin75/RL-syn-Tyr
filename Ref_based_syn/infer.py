from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from rlsyn_base.paths import data_path, ensure_dir, read_first_column_csv, read_smiles_lines
from rlsyn_base.paths import test_path

_IMPORT_ERROR: Optional[Exception]
try:
    # RDKit-dependent imports
    from .data_utils import Reaction
    from .decomposer import BreakMol
    from .utils import (
        get_morgen_fingerprint,
        get_properties,
        strip_dummy_atoms,
    )
    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover (env-dependent)
    Reaction = None  # type: ignore[assignment]
    BreakMol = None  # type: ignore[assignment]
    get_morgen_fingerprint = None  # type: ignore[assignment]
    get_properties = None  # type: ignore[assignment]
    strip_dummy_atoms = None  # type: ignore[assignment]
    _IMPORT_ERROR = e


FALLBACK_REF_SMILES = "CC1=CC=C(NC(=O)CCCN2CCN(C/C=C/C3=CC=CC=C3)CC2)C(C)=C1"


def default_ref_smiles() -> str:
    """
    Prefer the canonical test reference SMILES (AI10) if present:
      repo-root/test/2y9x/compounds.smi
    Otherwise fall back to the historical hard-coded SMILES.
    """
    p = Path(test_path("2y9x", "compounds.smi"))
    if p.exists():
        try:
            s = p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            if s:
                return s
        except Exception:
            pass
    return FALLBACK_REF_SMILES


@dataclass
class Candidate:
    product_smiles: str
    rxn_smirks: str
    reactant1: str
    reactant2: Optional[str]
    similarity_to_ref: float
    drug_like: bool
    docking_score: Optional[float]
    synthesizable: bool
    total_score: float


def tanimoto_from_bits(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.sum(a * b))
    denom = float(np.sum(a) + np.sum(b) - inter)
    return 0.0 if denom == 0.0 else inter / denom


def fp_bits(smiles: str, nbits: int) -> np.ndarray:
    return np.array(get_morgen_fingerprint(smiles, nBits=nbits), dtype=np.float32)


def decompose_query_frags(ref_smiles: str, *, lower_limit: int = 5) -> list[str]:
    """
    Decompose the reference molecule into multiple query fragments, as described in README.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(ref_smiles)
    if mol is None:
        raise ValueError(f"Invalid reference SMILES: {ref_smiles}")

    query_frags: list[str] = []
    for cut_num in (1, 2):
        bm = BreakMol(mol, lower_limit=lower_limit, cut_num=cut_num)
        for frags, _break_bonds, _ori_index in bm.enumerate_break():
            frags_smi = []
            for frag in frags:
                frag_smi = Chem.MolToSmiles(strip_dummy_atoms(frag))
                frags_smi.append(frag_smi)
            query_frags.extend(frags_smi)

    return sorted(set(query_frags))


def search_similar_building_blocks(
    query_frags: list[str],
    building_blocks: list[str],
    *,
    nbits: int = 256,
    threshold: float = 0.5,
    max_keep: int = 500,
) -> list[str]:
    """
    For each query fragment, retrieve structurally related fragments/building blocks by Tanimoto.
    This mirrors the dataset-building idea, but works with any provided building-block list.
    """
    if not building_blocks:
        return []

    bb_fps = np.stack([fp_bits(s, nbits) for s in building_blocks], axis=0)

    keep: set[str] = set()
    for frag in query_frags:
        fpf = fp_bits(frag, nbits)
        inter = np.sum(bb_fps * fpf[None, :], axis=1)
        denom = np.sum(bb_fps, axis=1) + np.sum(fpf) - inter
        sims = np.where(denom == 0.0, 0.0, inter / denom)
        idx = np.where(sims >= threshold)[0]
        for i in idx.tolist():
            keep.add(building_blocks[i])
            if len(keep) >= max_keep:
                return sorted(keep)
    return sorted(keep)


def load_reactions_from_file(path: str, *, max_templates: int) -> list[Reaction]:
    rxns: list[Reaction] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # template files sometimes have extra columns; keep first token
            smirks = s.split()[0]
            try:
                rxn = Reaction(smirks)
            except Exception:
                continue
            rxns.append(rxn)
            if len(rxns) >= max_templates:
                break
    return rxns


def try_make_docker(do_docking: bool):
    if not do_docking:
        return None
    try:
        from .dock import DockingVina, get_docking_config_for_vina

        cfg = get_docking_config_for_vina()
        # If receptor or vina is missing, skip docking cleanly.
        if not Path(cfg["vina_program"]).exists():
            return None
        if not Path(cfg["receptor_file"]).exists():
            return None
        return DockingVina(cfg)
    except Exception:
        return None


def generate_candidates(
    ref_smiles: str,
    *,
    building_blocks: list[str],
    template_file: str,
    max_templates: int = 100,
    max_bbs: int = 300,
    num_candidates: int = 128,
    seed: int = 0,
    do_docking: bool = False,
) -> tuple[list[Candidate], dict]:
    """
    A lightweight inference pipeline (no RL checkpoints required) that follows README logic:
      - fragment the reference molecule
      - retrieve similar building blocks
      - generate molecules via reaction templates
      - evaluate similarity + (optional) docking + drug-likeness
      - rank & return
    """
    random.seed(seed)
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing runtime dependencies for Ref_based_syn inference. "
            "Please install the Ref_based_syn conda environment (see Ref_based_syn/README.md). "
            f"Original import error: {_IMPORT_ERROR}"
        )

    query_frags = decompose_query_frags(ref_smiles)
    similar_bbs = search_similar_building_blocks(query_frags, building_blocks)
    if not similar_bbs:
        # Fallback: still allow a demo run with a random subset
        similar_bbs = building_blocks[:]

    random.shuffle(similar_bbs)
    similar_bbs = similar_bbs[:max_bbs]

    rxns = load_reactions_from_file(template_file, max_templates=max_templates)
    for rxn in rxns:
        rxn.set_available_reactants(similar_bbs)

    docker = try_make_docker(do_docking)

    fp_ref = fp_bits(ref_smiles, 2048)

    products: dict[str, tuple[Reaction, str, Optional[str]]] = {}
    # biased sampling towards bi-molecular (more diverse)
    rxn_pool = rxns[:]
    random.shuffle(rxn_pool)
    for rxn in rxn_pool:
        if len(products) >= num_candidates:
            break
        if rxn.num_reactant == 1:
            rlist = rxn.available_reactants[0]
            if not rlist:
                continue
            for _ in range(10):
                r1 = random.choice(rlist)
                prod = rxn.run_reaction([r1, None])
                if prod and prod not in products:
                    products[prod] = (rxn, r1, None)
                    break
        else:
            rlist1, rlist2 = rxn.available_reactants
            if not rlist1 or not rlist2:
                continue
            for _ in range(30):
                r1 = random.choice(rlist1)
                r2 = random.choice(rlist2)
                prod = rxn.run_reaction([r1, r2])
                if prod and prod not in products:
                    products[prod] = (rxn, r1, r2)
                    break

    candidates: list[Candidate] = []
    for prod, (rxn, r1, r2) in products.items():
        fp_prod = fp_bits(prod, 2048)
        sim = tanimoto_from_bits(fp_ref, fp_prod)
        drug_like = bool(get_properties(prod))

        docking_score: Optional[float] = None
        if docker is not None:
            try:
                docking_score = float(docker.predict([prod])[0])
            except Exception:
                docking_score = None

        synthesizable = True  # produced by a valid reaction template + available reactants

        # A simple integrative score (can be refined later):
        # - similarity: higher is better
        # - drug_like: bonus if passes basic filters
        # - docking: lower is better; convert to a small bonus when available
        docking_bonus = 0.0
        if docking_score is not None and np.isfinite(docking_score):
            docking_bonus = max(0.0, (10.0 - docking_score)) * 0.01
        total = sim + (0.05 if drug_like else 0.0) + docking_bonus

        candidates.append(
            Candidate(
                product_smiles=prod,
                rxn_smirks=rxn.smirks,
                reactant1=r1,
                reactant2=r2,
                similarity_to_ref=float(sim),
                drug_like=drug_like,
                docking_score=docking_score,
                synthesizable=synthesizable,
                total_score=float(total),
            )
        )

    candidates.sort(key=lambda c: c.total_score, reverse=True)
    meta = {
        "ref_smiles": ref_smiles,
        "query_frags_count": len(query_frags),
        "similar_bbs_count": len(similar_bbs),
        "rxn_templates_loaded": len(rxns),
        "docking_enabled": docker is not None,
        "note": "This is a lightweight inference/validation pipeline; RL checkpoint inference can be added when ckpt files are available.",
    }
    return candidates, meta


def run_validation(args: argparse.Namespace) -> int:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing runtime dependencies for Ref_based_syn inference. "
            "Please install the Ref_based_syn conda environment (see Ref_based_syn/README.md). "
            f"Original import error: {_IMPORT_ERROR}"
        )
    ref = args.ref_smiles or default_ref_smiles()

    # keep root-level data/ unchanged as requested
    bb_csv = args.building_blocks_csv or data_path("building_blocks_inland.csv")
    template_file = args.template_file or data_path("template_top100.csv")

    building_blocks = read_first_column_csv(bb_csv, limit=args.max_building_blocks)
    if not building_blocks:
        raise RuntimeError(f"Failed to load building blocks from: {bb_csv}")

    candidates, meta = generate_candidates(
        ref,
        building_blocks=building_blocks,
        template_file=template_file,
        max_templates=args.max_templates,
        max_bbs=args.max_bbs,
        num_candidates=args.num_candidates,
        seed=args.seed,
        do_docking=args.do_docking,
    )

    out_dir = ensure_dir(args.out_dir or (Path.cwd() / "results"))
    ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
    out_file = Path(out_dir) / f"ref_infer_validation_{ts}.json"

    payload = {
        "meta": meta,
        "top_k": [asdict(c) for c in candidates[: args.top_k]],
        "all_count": len(candidates),
    }
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[validate] ref={ref}")
    print(f"[validate] candidates={len(candidates)} top_k={min(args.top_k, len(candidates))}")
    print(f"[validate] output={out_file}")

    # basic verification signals
    if len(candidates) == 0:
        return 2
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ref_based_syn inference + validation")
    p.add_argument("--validate", action="store_true", help="Run built-in validation case")
    p.add_argument("--ref_smiles", type=str, default=None, help="Reference/lead compound SMILES; default reads test/2y9x/compounds.smi")
    p.add_argument("--building_blocks_csv", type=str, default=None)
    p.add_argument("--template_file", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_templates", type=int, default=50)
    p.add_argument("--max_building_blocks", type=int, default=2000)
    p.add_argument("--max_bbs", type=int, default=300)
    p.add_argument("--num_candidates", type=int, default=128)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--do_docking", action="store_true", help="Enable docking if qvina+receptor are available")
    return p


def main():
    args = build_argparser().parse_args()
    if args.validate:
        raise SystemExit(run_validation(args))

    # default to validation behavior for now (explicit is nicer)
    print("Use `--validate` to run the full inference+validation demo.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()


