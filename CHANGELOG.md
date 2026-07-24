# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial project scaffolding with Clean Architecture layers
- Domain models: User, FaceEncoding, RecognitionEvent
- Repository interfaces for persistence abstraction
- Face recognition engine interface and dlib-based implementation
- SQLite persistence layer
- OpenCV camera adapter
- Embedding storage service
- Use cases for register, capture, train, recognize, manage users, export logs
- Click-based CLI with all required commands
- Configuration management with environment variables
- Rich terminal output
- Comprehensive logging and log export (CSV/JSON)
- Test structure with pytest

### Changed

- N/A

### Deprecated

- N/A

### Removed

- N/A

### Fixed

- N/A

### Security

- N/A
