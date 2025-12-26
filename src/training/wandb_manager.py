"""WandB multi-run manager for budget-based logging."""
import wandb
from typing import Optional
from src.training.trainer import format_budget


class BudgetRunManager:
    """Logs to multiple WandB runs simultaneously, one per target budget.
    
    During full stage, logs to each budget's run only up to its branch point
    (not the full budget) to avoid duplicates when decay resumes.
    """
    
    def __init__(
        self, 
        project: str,
        group: str,
        base_config: dict,
        target_checkpoints: list[tuple[int, int]],  # (branch_tokens, budget)
        rank: int = 0,
        run_ids: dict[int, str] | None = None,
    ):
        self.project = project
        self.group = group
        self.base_config = base_config
        self.rank = rank
        self._resume_run_ids = run_ids or {}
        
        # store branch_tokens -> budget mapping and sorted budgets
        self.branch_points: dict[int, int] = {}  # budget -> branch_tokens
        self.target_budgets = []
        for branch_tokens, budget in sorted(target_checkpoints, key=lambda x: x[1]):
            self.target_budgets.append(budget)
            self.branch_points[budget] = branch_tokens
        
        # max budget logs all the way (full stage handles its own decay)
        self.max_budget = max(self.target_budgets) if self.target_budgets else 0
        
        # extract naming info from config
        self.model_size = base_config.get("model_size", "unknown")
        self.pe_type = base_config.get("pe_type", "unknown")
        self.seed = base_config.get("seed", 42)
        
        # budget -> Run object
        self.runs: dict[int, wandb.sdk.wandb_run.Run] = {}
        
        if rank == 0:
            self._init_all_runs()
    
    def _init_all_runs(self):
        """Create a WandB run for each budget."""
        for budget in self.target_budgets:
            budget_str = format_budget(budget)
            run_name = f"{self.model_size}_{self.pe_type}_{budget_str}_s{self.seed}"

            resume_id = self._resume_run_ids.get(budget)
            if resume_id:
                run = wandb.init(
                    project=self.project,
                    id=resume_id,
                    resume="must",
                    reinit=True,
                )
            else:
                run = wandb.init(
                    project=self.project,
                    group=self.group,
                    name=run_name,
                    config={**self.base_config, "target_budget": budget},
                    reinit=True,
                )
            self.runs[budget] = run
    
    def log(self, metrics: dict, consumed_tokens: int):
        """Log to budget runs up to their branch point (not full budget).
        
        Max budget run logs all the way since full stage handles its decay.
        """
        if self.rank != 0:
            return
        
        for budget, run in self.runs.items():
            # max budget logs all the way; others stop at branch point
            if budget == self.max_budget:
                cutoff = budget
            else:
                cutoff = self.branch_points.get(budget, budget)
            
            if consumed_tokens <= cutoff:
                run.log({**metrics, "tokens": consumed_tokens}, step=consumed_tokens)
    
    def get_run_id(self, budget: int) -> Optional[str]:
        """Get run ID for a specific budget."""
        if budget in self.runs:
            return self.runs[budget].id
        return None
    
    def get_all_run_ids(self) -> dict[int, str]:
        """Get all budget -> run_id mappings."""
        return {b: run.id for b, run in self.runs.items()}
    
    def finish_all(self):
        """Finish all runs."""
        if self.rank != 0:
            return
        for run in self.runs.values():
            run.finish()