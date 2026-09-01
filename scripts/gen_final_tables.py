"""
Deprecated compatibility wrapper.

Use make_results_tables.py. This wrapper is kept so old commands do not
regenerate simulated/estimated paper tables.
"""

from make_results_tables import main


if __name__ == "__main__":
    main()
