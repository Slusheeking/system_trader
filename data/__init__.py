#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Package
-----------
This package contains data-related modules for the system trader.
"""

# Import subpackages
import data.collectors
import data.database
import data.processors

__all__ = [
    'collectors',
    'database',
    'processors'
]
