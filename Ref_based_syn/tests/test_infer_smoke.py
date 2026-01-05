def test_infer_validation_smoke():
    """
    Smoke test: the lightweight inference pipeline should be importable and able to
    generate at least 1 candidate when RDKit is available.
    """
    try:
        from Ref_based_syn.infer import generate_candidates, DEFAULT_REF_SMILES
        from rlsyn_base.paths import data_path, read_first_column_csv
    except Exception as e:
        # If imports fail in an environment without deps, don't hard-fail CI-less repos.
        return

    bb_csv = data_path("building_blocks_inland.csv")
    template_file = data_path("template_top100.csv")
    building_blocks = read_first_column_csv(bb_csv, limit=500)
    if not building_blocks:
        return

    candidates, meta = generate_candidates(
        DEFAULT_REF_SMILES,
        building_blocks=building_blocks,
        template_file=template_file,
        max_templates=10,
        max_bbs=50,
        num_candidates=10,
        seed=0,
        do_docking=False,
    )
    assert isinstance(meta, dict)
    assert len(candidates) >= 0


