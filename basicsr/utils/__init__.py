"""Inference-only package init.

The upstream module imports training utilities that this project does not vendor.
Checkpoint loading imports ``basicsr.utils.registry`` directly.
"""
