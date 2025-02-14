#!/bin/bash
set -eo

uv sync
git add -u uv.lock
