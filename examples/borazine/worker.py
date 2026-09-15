import sys
import json
import logging
import time
from dftscape.core.search_primitives import RunCondition
from dftscape.core.point import CriticalPoint
from dftscape.backends import BackendRegistry
from dftscape.core.dynamics import HiSD
from dftscape.core.metrics import MetricRegistry
from dftscape.core.subspace_handlers import SubspaceHandler
from dftscape.core.symmetry_handlers import SymmetryRegistry
from dftscape.core.eigensolvers import EigensolverRegistry
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_hisd_runner(backend, cfg):
    """Rebuilds the exact HiSD runner from the dictionary config."""
    metrics = [MetricRegistry.create_metric(m) for m in cfg['hisd']['metrics']]
    symmetry_handler = SymmetryRegistry.create(cfg['hisd']['symmetry_handler'])

    eigensolver_kwargs = {
        "backend": backend,
        "k": cfg['eigensolver']['k'],
        "tol": cfg['eigensolver']['tol'],
        "maxiter": cfg['eigensolver']['maxiter'],
    }
    eigensolver = EigensolverRegistry.create(cfg['eigensolver']['name'], **eigensolver_kwargs)

    subspace_handler = SubspaceHandler(
        backend=backend, symmetry_handler=symmetry_handler, eigensolver=eigensolver
    )

    hisd_kwargs = {
        "backend": backend,
        "subspace_handler": subspace_handler,
        "symmetry_handler": symmetry_handler,
        "dt": cfg['hisd']['dt'],
        "maxiter": cfg['hisd']['maxiter'],
        "grad_norm_tol": cfg['hisd']['grad_norm_tol'],
        "log_interval": cfg['hisd']['log_interval'],
        "subspace_update_interval": cfg['hisd']['subspace_update_interval'],
        "metrics": metrics
    }
    if cfg['hisd'].get("energy_threshold") is not None:
        hisd_kwargs["energy_threshold"] = cfg['hisd']["energy_threshold"]

    return HiSD(**hisd_kwargs)

def main():
    if len(sys.argv) != 3:
        print("Usage: python worker.py <input.json> <output.json>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_json = sys.argv[2]

    # Load the job package
    with open(input_json, 'r') as f:
        job_data = json.load(f)

    cfg = job_data['cfg']
    run_cond = RunCondition.from_config(job_data['run_condition'])
    parent_xyz = job_data['parent_xyz']
    start_time = time.time()
    result_payload = {"success": False, "error": None, "critical_point": None, "execution_time": 0.0, "add_edge": False}

    try:
        # Build backend
        backend_kwargs = {k: v for k, v in cfg['backend'].items() if k != "name"}
        # Feed the exact XYZ string to the backend so it knows the elements!
        # HiSD will still use run_cond.state automatically when we call hisd_runner.run()
        backend_kwargs["molecule_xyz"] = parent_xyz
        backend = BackendRegistry.create(cfg['backend']['name'], **backend_kwargs)
        
        # Build runner and execute
        hisd_runner = create_hisd_runner(backend, cfg)
        cp, add_edge = hisd_runner.run(run_cond)
        
        if cp is not None:
            result_payload["success"] = True
            result_payload["critical_point"] = cp.to_config()
            result_payload["add_edge"] = add_edge
        else:
            result_payload["error"] = "Optimization failed to converge."

    except Exception as e:
        logger.exception("Worker crashed during execution.")
        result_payload["error"] = str(e)

    result_payload["execution_time"] = time.time() - start_time

    # Safely dump the results for the orchestrator to harvest
    with open(output_json, 'w') as f:
        json.dump(result_payload, f, indent=2)

if __name__ == "__main__":
    main()
