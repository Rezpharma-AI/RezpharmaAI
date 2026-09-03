@echo off
title RezpharmaCDSS - Tests
call rezpharma_env\Scripts\activate.bat
pytest tests/ -v --tb=short
pause
