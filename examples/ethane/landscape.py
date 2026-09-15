from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path

from dftscape.backends import BackendRegistry
from dftscape.core.dynamics import HiSD
from dftscape.core.point import CriticalPoint
from dftscape.core.search_orchestrator import (
    SearchOrchestrator,
    SearchOrchestratorRegistry,
)
from dftscape.core.callbacks import SearchCheckpointerCallback, SearchInfoCallback
from dftscape.core.graph import SearchableCriticalPointGraph
from dftscape.core.metrics import MetricRegistry
from dftscape.checkpoint import CheckpointRegistry
from dftscape.core.subspace_handlers import SubspaceHandler
from dftscape.core.symmetry_handlers import SymmetryRegistry
from dftscape.core.eigensolvers import EigensolverRegistry
from dftscape.backends import Backend

logger = logging.getLogger(__name__)


def load_initial_point(filepath: str) -> CriticalPoint:
    logger.info(f"Loading initial point from {filepath}")

    # Create JSON checkpointer
    checkpointer = CheckpointRegistry.create("json")

    # Load the payload from JSON file
    payload = checkpointer.load(filepath)

    # Create CriticalPoint from config
    initial_point = CriticalPoint.from_config(payload)

    return initial_point


def create_hisd_runner(backend: Backend, cfg: DictConfig) -> HiSD:
    # Create metrics from config
    metrics = [MetricRegistry.create_metric(m) for m in cfg.hisd.metrics]

    symmetry_handler = SymmetryRegistry.create(cfg.hisd.symmetry_handler)

    eigensolver_kwargs = {
        "backend": backend,
        "k": cfg.eigensolver.k,
        "tol": cfg.eigensolver.tol,
        "maxiter": cfg.eigensolver.maxiter,
        #"largest": cfg.eigensolver.largest,
        #"random_state": cfg.eigensolver.random_state,
    }
    eigensolver = EigensolverRegistry.create(cfg.eigensolver.name, **eigensolver_kwargs)

    subspace_handler = SubspaceHandler(
        backend=backend, symmetry_handler=symmetry_handler, eigensolver=eigensolver
    )

    hisd_kwargs = {
        "backend": backend,
        "subspace_handler": subspace_handler,
        "symmetry_handler": symmetry_handler,
        "dt": cfg.hisd.dt,
        "maxiter": cfg.hisd.maxiter,
        "grad_norm_tol": cfg.hisd.grad_norm_tol,
        "log_interval": cfg.hisd.log_interval,
        "subspace_update_interval": cfg.hisd.subspace_update_interval,
        "metrics": metrics
    }

    # Only add energy_threshold if it's specified in config
    if hasattr(cfg.hisd, "energy_threshold") and cfg.hisd.energy_threshold is not None:
        hisd_kwargs["energy_threshold"] = cfg.hisd.energy_threshold

    hisd_runner = HiSD(**hisd_kwargs)

    return hisd_runner


def log_checkpoint_summary(
    checkpoint_callback: SearchCheckpointerCallback, cfg: DictConfig, logger
) -> None:
    if not cfg.periodic_checkpoints.enabled:
        return

    checkpoint_count, checkpoint_names = checkpoint_callback.get_checkpoint_summary()
    if checkpoint_count > 0:
        logger.info(f"Periodic checkpoints saved: {checkpoint_count}")
        for name in checkpoint_names:
            logger.info(f"  - {name}")
    else:
        logger.info(
            "No periodic checkpoints were saved (search completed too quickly or no jobs processed)"
        )


def create_graph_and_orchestrator(
    cfg: DictConfig, backend: Backend, hisd_runner: HiSD, initial_point: CriticalPoint, run_dir: str
) -> tuple[SearchableCriticalPointGraph, SearchOrchestrator]:
    # Set up graph
    graph_kwargs = {
        "threshold": cfg.graph.threshold,
        "distance_function": cfg.graph.distance_function,
    }

    # Only add align if it's explicitly set in config
    if hasattr(cfg.graph, "align"):
        graph_kwargs["align"] = cfg.graph.align

    graph = SearchableCriticalPointGraph(**graph_kwargs)

    # Set up orchestrator
    from dftscape.core.job_queues import JobQueues
    job_queues = JobQueues()
    
    # We build the orchestrator kwargs dynamically so we can inject
    # parallel-specific args if needed
    orchestrator_kwargs = {
        "graph": graph,
        "job_queues": job_queues,
        "hisd_runner": hisd_runner,
        "d_max": cfg.orchestrator.get("d_max", None),
        "n_directions": cfg.orchestrator.get("n_directions", 1),
        "perturbation_magnitude": cfg.orchestrator.perturbation_magnitude,
        "callbacks": None,  # Will set after creation
    }

    # Inject parallel-specific configs if the user requested the parallel orchestrator
    if cfg.orchestrator.type == "parallel":
        # Convert OmegaConf to standard dict for the JSON serialization in worker
        orchestrator_kwargs["base_cfg"] = OmegaConf.to_container(cfg, resolve=True)
        # Add output specific directory for the PBS scripts
        orchestrator_kwargs["job_dir"] = Path(run_dir) / "pbs_jobs"
        if "max_concurrent" in cfg.orchestrator:
            orchestrator_kwargs["max_concurrent"] = cfg.orchestrator.max_concurrent

    orchestrator = SearchOrchestratorRegistry.create(
        cfg.orchestrator.type,
        **orchestrator_kwargs
    )

    # Add initial point to graph
    graph.add(initial_point)

    return graph, orchestrator


@hydra.main(version_base=None, config_path="config", config_name="landscape")
def search_landscape(cfg: DictConfig) -> None:

    logger.info("=== Ammonia borane Landscape Search ===\n")
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Load initial critical point first to get molecule_xyz
    logger.info("Loading initial critical point...")
    data_path = Path(__file__).parent / cfg.initial_point.data_path
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    initial_point = load_initial_point(str(data_path))
    logger.info(f"   ✓ Loaded point with Morse index {initial_point.index}")
    logger.info(f"   Energy: {initial_point.energy:.6f}")

    # Set up backend from config
    logger.info(f"\nSetting up {cfg.backend.name} backend...")
    backend_kwargs = {k: v for k, v in cfg.backend.items() if k != "name"}
    backend_kwargs["molecule_xyz"] = initial_point.xyz
    backend = BackendRegistry.create(cfg.backend.name, **backend_kwargs)
    logger.info(f"   ✓ {cfg.backend.name.title()}Backend created")

    hisd_runner = create_hisd_runner(backend, cfg)
    checkpoint_dir = cfg.get("checkpoint_dir", None)

    # Set up checkpointer
    checkpoint_callback = SearchCheckpointerCallback(
        output_dir=run_dir,
        enabled=cfg.periodic_checkpoints.enabled,
        interval=cfg.periodic_checkpoints.interval,
        max_count=cfg.periodic_checkpoints.get("max_count", 20),
        prefix=cfg.periodic_checkpoints.get("prefix", "checkpoint_periodic"),
    )

    if checkpoint_dir is not None and checkpoint_dir != "null":
        logger.info(f"Loading search state from checkpoint: {checkpoint_dir}")
        graph, job_queues = checkpoint_callback.load_checkpoint(checkpoint_dir)
        
        # Build kwargs for reloading, injecting parallel info if needed
        orchestrator_kwargs = {
            "graph": graph,
            "job_queues": job_queues,
            "hisd_runner": hisd_runner,
            "callbacks": [checkpoint_callback],
            "d_max": cfg.orchestrator.get("d_max", None),
            "n_directions": cfg.orchestrator.get("n_directions", 1),
            "perturbation_magnitude": cfg.orchestrator.perturbation_magnitude,
        }
        
        if cfg.orchestrator.type == "parallel":
            orchestrator_kwargs["base_cfg"] = OmegaConf.to_container(cfg, resolve=True)
            orchestrator_kwargs["job_dir"] = Path(run_dir) / "pbs_jobs"
            if "max_concurrent" in cfg.orchestrator:
                orchestrator_kwargs["max_concurrent"] = cfg.orchestrator.max_concurrent
                
        # Recreate using the specific registry type!
        orchestrator = SearchOrchestratorRegistry.create(
            cfg.orchestrator.type,
            **orchestrator_kwargs
        )
        
        # Reconstruct JobTracker state from the loaded job queues
        orchestrator.job_tracker.reconstruct_from_job_queues(job_queues)
        logger.info("Reconstructed JobTracker state from checkpoint")
    else:
        logger.info("Starting fresh search (no checkpoint specified)")
        graph, orchestrator = create_graph_and_orchestrator(
            cfg, backend, hisd_runner, initial_point, run_dir
        )

    # Set up callbacks
    searchinfo_callback = SearchInfoCallback()
    # If resuming, the checkpoint callback is already in the list, so just append searchinfo
    if orchestrator.callbacks is None:
        orchestrator.callbacks = [checkpoint_callback, searchinfo_callback]
    elif searchinfo_callback not in orchestrator.callbacks:
        orchestrator.callbacks.append(searchinfo_callback)

    # Initial report
    logger.info(orchestrator.report())

    # ---------------------------------------------------------
    # EXHAUSTIVE SEARCH LOOP
    # ---------------------------------------------------------
    logger.info("Starting exhaustive crawler search (Upward <-> Downward)...")
    cycle = 1
    
    try:
        # Loop until BOTH queues are completely empty and all nodes are processed
        while not (orchestrator.is_upward_search_complete() and orchestrator.is_downward_search_complete()):
            logger.info(f"\n=== Starting Search Cycle {cycle} ===")

            # reset the checkpoint callback's internal tracker for the new cycle
            checkpoint_callback.last_checkpoint_job = 0
            
            # exhaust all currently known upward paths
            if not orchestrator.is_upward_search_complete():
                logger.info(f"Cycle {cycle}: Running upward search phase...")
                orchestrator.run_upward_search()
                logger.info(f"Cycle {cycle}: Upward phase complete.")
                checkpoint_callback.trigger_manual_checkpoint(orchestrator, suffix=f"upward_cycle_{cycle}")

            # exhaust all downward paths (This will likely discover new upward paths!)
            if not orchestrator.is_downward_search_complete():
                logger.info(f"Cycle {cycle}: Running downward search phase...")
                orchestrator.run_downward_search()
                logger.info(f"Cycle {cycle}: Downward phase complete.")
                checkpoint_callback.trigger_manual_checkpoint(orchestrator, suffix=f"downward_cycle_{cycle}")
                
            cycle += 1
            
        logger.info("\n✓ Exhaustive search complete! Entire network has been mapped.")
    except Exception as e:
        logger.error(f"  ✗ Search failed during cycle {cycle}: {e}")
        import traceback
        traceback.print_exc()
        return
    # ---------------------------------------------------------

    checkpoint_callback.trigger_manual_checkpoint(orchestrator, suffix="final")

    # Final summary
    logger.info(orchestrator.report())

    # Log periodic checkpoint summary
    log_checkpoint_summary(checkpoint_callback, cfg, logger)

    logger.info(f"Output directory: {run_dir}")


if __name__ == "__main__":
    search_landscape()
