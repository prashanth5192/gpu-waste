from gpu_waste_meter import _env

VARS = ["GWM_JOB", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "GWM_ATTEMPT",
        "TORCHELASTIC_RESTART_COUNT", "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS"]


def clean(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)


def test_job_precedence(monkeypatch):
    clean(monkeypatch)
    assert _env.job_id().startswith("job-")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert _env.job_id() == "123"
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "te")
    assert _env.job_id() == "te"
    monkeypatch.setenv("GWM_JOB", "mine")
    assert _env.job_id() == "mine"


def test_hash_is_stable(monkeypatch):
    clean(monkeypatch)
    assert _env.job_id() == _env.job_id()


def test_attempt(monkeypatch):
    clean(monkeypatch)
    assert _env.attempt_id() == 0
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "2")
    assert _env.attempt_id() == 2
    monkeypatch.setenv("GWM_ATTEMPT", "5")
    assert _env.attempt_id() == 5


def test_torchrun_identity(monkeypatch):
    clean(monkeypatch)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    i = _env.get_identity()
    assert (i["rank"], i["local_rank"], i["world_size"]) == (3, 1, 8)


def test_slurm_identity(monkeypatch):
    clean(monkeypatch)
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    i = _env.get_identity()
    assert (i["rank"], i["local_rank"], i["world_size"]) == (5, 1, 16)


def test_no_env_single_process(monkeypatch):
    clean(monkeypatch)
    i = _env.get_identity()
    assert i["rank"] is None and i["world_size"] is None and i["node"]
