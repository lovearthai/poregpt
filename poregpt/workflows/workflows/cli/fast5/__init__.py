# nanopore_llm_workflow/cli/fast5/__init__.py
# 暴露 main 函数（可选，但推荐）
# 这样你也可以在 Python 中写：
# from nanopore_llm_workflow.cli.fast5 import main
# main()
from .fast5_to_chunks import main
__all__ = ["main"]
