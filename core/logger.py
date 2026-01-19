"""Unified logging system for the entire pipeline."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class PipelineLogger:
    """Unified logger for complete pipeline execution with structured sections."""
    
    def __init__(self, log_dir: Path = Path("logs"), run_id: Optional[str] = None):
        """Initialize unified logger.
        
        Args:
            log_dir: Directory for log files
            run_id: Unique identifier for this run (default: timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_id = run_id
        self.log_file = self.log_dir / f"pipeline_{run_id}.log"
        
        # Setup logger
        self.logger = logging.getLogger(f"pipeline_{run_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # Detailed formatter for file
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Simpler formatter for console
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler (detailed logging)
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler (info and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Track metrics
        self.metrics = {}
        
    def section(self, title: str, level: int = 1):
        """Log a section header.
        
        Args:
            title: Section title
            level: Header level (1=main, 2=sub, 3=detail)
        """
        if level == 1:
            separator = "=" * 80
            self.logger.info(separator)
            self.logger.info(f"  {title}")
            self.logger.info(separator)
        elif level == 2:
            separator = "-" * 60
            self.logger.info(separator)
            self.logger.info(f" {title}")
            self.logger.info(separator)
        else:
            self.logger.info(f"• {title}")
    
    def dataset_start(self, dataset_name: str, dataset_num: int, total_datasets: int):
        """Log start of dataset processing.
        
        Args:
            dataset_name: Name of dataset
            dataset_num: Current dataset number
            total_datasets: Total number of datasets
        """
        self.section(f"DATASET {dataset_num}/{total_datasets}: {dataset_name}", level=1)
    
    def phase_start(self, phase_name: str):
        """Log start of a pipeline phase.
        
        Args:
            phase_name: Name of the phase
        """
        self.section(phase_name, level=2)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def metric(self, dataset: str, phase: str, metrics: dict):
        """Log metrics for a specific phase and dataset.
        
        Args:
            dataset: Dataset name
            phase: Phase name
            metrics: Dictionary of metrics
        """
        key = f"{dataset}_{phase}"
        self.metrics[key] = metrics
        
        # Log metrics in readable format
        self.logger.info(f"  Metrics:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"    {metric_name}: {value:.4f}")
            elif isinstance(value, int):
                self.logger.info(f"    {metric_name}: {value:,}")
            else:
                self.logger.info(f"    {metric_name}: {value}")
    
    def summary_table(self, data: list, headers: list):
        """Log a summary table.
        
        Args:
            data: List of lists (rows)
            headers: List of column headers
        """
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Format header
        header_line = " | ".join(
            str(h).ljust(w) for h, w in zip(headers, col_widths)
        )
        separator = "-+-".join("-" * w for w in col_widths)
        
        self.logger.info(header_line)
        self.logger.info(separator)
        
        # Format data rows
        for row in data:
            row_line = " | ".join(
                str(cell).ljust(w) for cell, w in zip(row, col_widths)
            )
            self.logger.info(row_line)
    
    def pipeline_summary(self, dataset_results: dict, total_time: float):
        """Log final pipeline summary.
        
        Args:
            dataset_results: Dict mapping dataset name to results dict
            total_time: Total execution time in seconds
        """
        self.section("PIPELINE EXECUTION SUMMARY", level=1)
        
        # Success count
        n_success = sum(1 for r in dataset_results.values() if r.get('success', False))
        n_total = len(dataset_results)
        
        self.logger.info(f"Datasets processed: {n_success}/{n_total} successful")
        self.logger.info(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        self.logger.info("")
        
        # Per-dataset summary table
        table_data = []
        for dataset, results in dataset_results.items():
            if results.get('success', False):
                table_data.append([
                    dataset,
                    "✓",
                    f"{results.get('n_species', 0)}",
                    f"{results.get('n_metabolites', 0)}",
                    f"{results.get('n_associations', 0):,}",
                    f"{results.get('n_modules', 0)}",
                    f"{results.get('execution_time', 0):.1f}s"
                ])
            else:
                table_data.append([
                    dataset,
                    "✗",
                    "-",
                    "-",
                    "-",
                    "-",
                    f"{results.get('execution_time', 0):.1f}s"
                ])
        
        headers = ["Dataset", "Status", "Species", "Metabolites", "Associations", "Modules", "Time"]
        self.summary_table(table_data, headers)
        
        self.logger.info("")
        if n_success == n_total:
            self.logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        else:
            self.logger.warning(f"⚠️  PIPELINE COMPLETED WITH {n_total - n_success} FAILURES")
        
        self.logger.info(f"Full log: {self.log_file}")
    
    def get_logger(self):
        """Get the underlying logger object."""
        return self.logger


def setup_logger(log_dir: Path = Path("logs"), run_id: Optional[str] = None) -> PipelineLogger:
    """Setup and return a pipeline logger.
    
    Args:
        log_dir: Directory for log files
        run_id: Unique identifier for this run
        
    Returns:
        PipelineLogger instance
    """
    return PipelineLogger(log_dir, run_id)
